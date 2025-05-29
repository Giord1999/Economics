import subprocess
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import os
import warnings

# TODO: inserire la struttura del capitale se necessario

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

# Load the tokenizer and model, and set the pad_token to eos_token
tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
model = AutoModelForCausalLM.from_pretrained('distilgpt2')
tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Initialize Hugging Face Transformers text generation pipeline
device = 0 if torch.cuda.is_available() else -1
generator = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    pad_token_id=tokenizer.eos_token_id,
    device=device
)

# -----------------------------
# Simulation parameters
# -----------------------------
num_buyers = 10
num_sellers = 10
num_rounds = 500

# Buyers and sellers preferences (utility and cost)
buyers_utility = np.random.uniform(100, 200, num_buyers)
sellers_cost = np.random.uniform(50, 150, num_sellers)

# Q-learning parameters
# State size is now 4: [utility (or cost), reservation_price, bid/ask price, beta]
state_size = 4
action_size = 3        # Actions: 0 = Increase, 1 = Decrease, 2 = Maintain

learning_rate = 0.001  # Learning rate for optimizers

# Exploration (epsilon-greedy)
initial_epsilon = 0.5
epsilon_decay = 0.99
min_epsilon = 0.1

# Frequency for LLM advice (every N rounds)
advice_frequency = 50

# Imitation learning probability: with this probability, an agent imitates another
imitation_probability = 0.1

# -----------------------------
# Helper functions
# -----------------------------
def normalize_state(state_tensor: torch.Tensor) -> torch.Tensor:
    """
    Normalize a state tensor by its maximum element to keep values in [0,1].
    Avoid division by zero by adding a small epsilon.
    """
    eps = 1e-6
    return state_tensor / (torch.max(state_tensor) + eps)

# -----------------------------
# Q-Network definition
# -----------------------------
class QNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# -----------------------------
# Agent classes
# -----------------------------
class Buyer:
    def __init__(self, utility: float, index: int):
        self.utility = utility
        # Reservation price drawn around 3-7.5 euros
        self.reservation_price = random.uniform(3.0, 7.5)
        # Initial bid price
        self.bid_price = max(self.reservation_price, utility * random.uniform(0.6, 1.0))
        # Index to identify cluster membership
        self.index = index
        # Time preference: beta_i in (0.85, 0.99)
        self.beta = random.uniform(0.85, 0.99)

    def get_state(self) -> torch.Tensor:
        """
        Return normalized state tensor:
        [utility, reservation_price, bid_price, beta].
        """
        raw_state = torch.tensor([
            self.utility,
            self.reservation_price,
            self.bid_price,
            self.beta
        ], dtype=torch.float32)
        return normalize_state(raw_state)

    def update_bid_rl(self, action: int):
        """
        Update bid_price according to Q-learning action.
        """
        if action == 0:  # Increase price
            self.bid_price *= random.uniform(1.05, 1.1)
        elif action == 1:  # Decrease price
            self.bid_price *= random.uniform(0.9, 0.95)
        # Ensure boundaries
        self.bid_price = max(self.bid_price, self.reservation_price)
        self.bid_price = min(self.bid_price, self.reservation_price * 2)

    def imitate_peer(self, peer_bid: float):
        """
        With some probability, adopt a peer's bid price.
        """
        self.bid_price = peer_bid

class Seller:
    def __init__(self, cost: float, index: int):
        self.cost = cost
        # Reservation price drawn around 3-7.5 euros
        self.reservation_price = random.uniform(3.0, 7.5)
        # Initial ask price
        self.ask_price = max(self.reservation_price, cost * random.uniform(1.1, 1.5))
        # Index to identify cluster membership
        self.index = index
        # Time preference: beta_i in (0.85, 0.99)
        self.beta = random.uniform(0.85, 0.99)

    def get_state(self) -> torch.Tensor:
        """
        Return normalized state tensor:
        [cost, reservation_price, ask_price, beta].
        """
        raw_state = torch.tensor([
            self.cost,
            self.reservation_price,
            self.ask_price,
            self.beta
        ], dtype=torch.float32)
        return normalize_state(raw_state)

    def update_price_rl(self, action: int):
        """
        Update ask_price according to Q-learning action.
        """
        if action == 0:  # Increase price
            self.ask_price *= random.uniform(1.05, 1.1)
        elif action == 1:  # Decrease price
            self.ask_price *= random.uniform(0.9, 0.95)
        # Ensure boundaries
        self.ask_price = max(self.ask_price, self.reservation_price)
        self.ask_price = min(self.ask_price, self.reservation_price * 2)

    def imitate_peer(self, peer_ask: float):
        """
        With some probability, adopt a peer's ask price.
        """
        self.ask_price = peer_ask

# -----------------------------
# LLM-based advice system
# -----------------------------
def get_cluster_index(agent_index: int, total_agents: int) -> int:
    """
    Determine cluster index (0 or 1) based on agent index.
    First half indices -> 0, second half -> 1.
    """
    midpoint = total_agents // 2
    return 0 if agent_index < midpoint else 1

# Advice cache keyed by (role, cluster_index, last_feedback)
# role: "buyer" or "seller"; cluster_index: 0 or 1; last_feedback: "success" or "failure"
advice_cache = {}

def llm_advice(role: str, feedback: str, reservation_price: float, cluster: int, round_num: int) -> float:
    """
    Generate or retrieve cached advice for a given (role, cluster, feedback).
    The advice is returned as a multiplicative factor for price adjustment.
    """
    global advice_cache

    cache_key = (role, cluster, feedback)
    # Refresh advice every 'advice_frequency' rounds or if not present
    if (round_num % advice_frequency == 0) or (cache_key not in advice_cache):
        prompt = (
            f"In cluster {cluster}, a {role} had a '{feedback}' transaction. "
            f"Their reservation price is {reservation_price:.2f} euros. "
            "What should they do to adjust their price in the next round?"
        )
        try:
            response = generator(
                prompt,
                max_new_tokens=50,     # ← uso solo max_new_tokens, non max_length
                num_return_sequences=1,
                truncation=True,
                clean_up_tokenization_spaces=True
            )
            advice_text = response[0]['generated_text'].strip().lower()
            advice_cache[cache_key] = advice_text
        except Exception as e:
            print(f"(LLM) Error generating advice for {cache_key}: {e}")
            advice_cache[cache_key] = "no change"

    advice_text = advice_cache[cache_key]
    # Interpret advice text into a multiplicative factor
    if "increase" in advice_text:
        return random.uniform(1.1, 1.2)
    elif "decrease" in advice_text:
        return random.uniform(0.85, 0.95)
    else:
        return random.uniform(0.95, 1.05)

# -----------------------------
# Initialize agents
# -----------------------------
buyers_advice = [Buyer(utility, idx) for idx, utility in enumerate(buyers_utility)]
sellers_advice = [Seller(cost, idx) for idx, cost in enumerate(sellers_cost)]

buyers_no_advice = [Buyer(utility, idx) for idx, utility in enumerate(buyers_utility)]
sellers_no_advice = [Seller(cost, idx) for idx, cost in enumerate(sellers_cost)]

# -----------------------------
# Initialize Q-networks and optimizers
# -----------------------------
q_network_advice = QNetwork(state_size, action_size)
q_network_no_advice = QNetwork(state_size, action_size)
optimizer_advice = optim.Adam(q_network_advice.parameters(), lr=learning_rate)
optimizer_no_advice = optim.Adam(q_network_no_advice.parameters(), lr=learning_rate)

# -----------------------------
# Metrics and logging
# -----------------------------
transactions_per_round_advice = []
transactions_per_round_no_advice = []
profits_over_time_advice = []
profits_over_time_no_advice = []
utilities_over_time_advice = []
utilities_over_time_no_advice = []
overall_bid_prices_advice = []
overall_ask_prices_advice = []
overall_bid_prices_no_advice = []
overall_ask_prices_no_advice = []
bid_ask_spread_advice = []
bid_ask_spread_no_advice = []
epsilon_values = []
advice_log = []

# -----------------------------
# Simulation loop
# -----------------------------
epsilon = initial_epsilon
for round_num in range(num_rounds):
    # Shuffle to randomize matches each round
    random.shuffle(buyers_advice)
    random.shuffle(sellers_advice)
    random.shuffle(buyers_no_advice)
    random.shuffle(sellers_no_advice)

    transactions_advice = 0
    transactions_no_advice = 0
    round_profit_advice = 0.0
    round_profit_no_advice = 0.0
    round_utility_advice = 0.0
    round_utility_no_advice = 0.0

    bid_prices_round_advice = []
    ask_prices_round_advice = []
    bid_prices_round_no_advice = []
    ask_prices_round_no_advice = []

    # -----------------------------
    # Simulation WITH LLM-based advice
    # -----------------------------
    for buyer, seller in zip(buyers_advice, sellers_advice):
        # Determine clusters
        cluster_buyer = get_cluster_index(buyer.index, num_buyers)
        cluster_seller = get_cluster_index(seller.index, num_sellers)

        # Get current states
        state_buyer = buyer.get_state()
        state_seller = seller.get_state()

        # ε-greedy action selection
        if random.random() > epsilon:
            with torch.no_grad():
                action_buyer = q_network_advice(state_buyer).argmax().item()
                action_seller = q_network_advice(state_seller).argmax().item()
        else:
            action_buyer = random.choice(range(action_size))
            action_seller = random.choice(range(action_size))

        # Perform RL-based price update
        buyer.update_bid_rl(action_buyer)
        seller.update_price_rl(action_seller)

        # Determine if a transaction occurs
        if buyer.bid_price >= seller.ask_price:
            raw_profit = max(0.0, buyer.utility - seller.cost)
            raw_utility = max(0.0, buyer.utility - buyer.bid_price)

            # Apply each agent's time preference (beta) to the raw payoff
            profit_buyer = raw_utility * buyer.beta
            profit_seller = raw_profit * seller.beta

            transactions_advice += 1
            round_profit_advice += profit_seller
            round_utility_advice += profit_buyer

            feedback_buyer = "success"
            feedback_seller = "success"
        else:
            profit_buyer = -1.0 * buyer.beta
            profit_seller = -1.0 * seller.beta
            feedback_buyer = "failure"
            feedback_seller = "failure"

        # Retrieve LLM advice factors
        factor_buyer = llm_advice("buyer", feedback_buyer, buyer.reservation_price, cluster_buyer, round_num)
        factor_seller = llm_advice("seller", feedback_seller, seller.reservation_price, cluster_seller, round_num)

        # Apply advice: adjust bid/ask prices
        buyer.bid_price *= factor_buyer
        seller.ask_price *= factor_seller

        # Re-enforce boundaries after advice
        buyer.bid_price = max(buyer.bid_price, buyer.reservation_price)
        buyer.bid_price = min(buyer.bid_price, buyer.reservation_price * 2)
        seller.ask_price = max(seller.ask_price, seller.reservation_price)
        seller.ask_price = min(seller.ask_price, seller.reservation_price * 2)

        # Log the advice
        advice_log.append(
            f"Round {round_num}: Buyer {buyer.index} ({feedback_buyer}) "
            f"advice factor {factor_buyer:.3f}; "
            f"Seller {seller.index} ({feedback_seller}) "
            f"advice factor {factor_seller:.3f}"
        )

        # Store prices for metrics
        bid_prices_round_advice.append(buyer.bid_price)
        ask_prices_round_advice.append(seller.ask_price)

        # Q-learning update every 10 rounds, using each agent's beta as discount factor
        if round_num % 10 == 0:
            next_state_buyer = buyer.get_state()
            next_state_seller = seller.get_state()
            with torch.no_grad():
                target_buyer = profit_buyer + buyer.beta * q_network_advice(next_state_buyer).max().item()
                target_seller = profit_seller + seller.beta * q_network_advice(next_state_seller).max().item()

            pred_buyer = q_network_advice(state_buyer)[action_buyer]
            pred_seller = q_network_advice(state_seller)[action_seller]
            loss_buyer = F.mse_loss(pred_buyer, torch.tensor(target_buyer, dtype=torch.float32))
            loss_seller = F.mse_loss(pred_seller, torch.tensor(target_seller, dtype=torch.float32))

            optimizer_advice.zero_grad()
            (loss_buyer + loss_seller).backward()
            optimizer_advice.step()

    # After all pairings, allow imitation among peers within each group
    for group_agents, is_buyer_group in [(buyers_advice, True), (sellers_advice, False)]:
        for agent in group_agents:
            if random.random() < imitation_probability:
                peer = random.choice([a for a in group_agents if a.index != agent.index])
                if is_buyer_group:
                    agent.imitate_peer(peer.bid_price)
                else:
                    agent.imitate_peer(peer.ask_price)

    # Record metrics for the "advice" simulation
    transactions_per_round_advice.append(transactions_advice)
    profits_over_time_advice.append(round_profit_advice)
    utilities_over_time_advice.append(round_utility_advice)
    overall_bid_prices_advice.append(np.mean(bid_prices_round_advice) if bid_prices_round_advice else 0.0)
    overall_ask_prices_advice.append(np.mean(ask_prices_round_advice) if ask_prices_round_advice else 0.0)
    bid_ask_spread_advice.append(
        np.mean(np.array(ask_prices_round_advice) - np.array(bid_prices_round_advice))
        if bid_prices_round_advice else 0.0
    )

    # -----------------------------
    # Simulation WITHOUT LLM-based advice
    # -----------------------------
    for buyer, seller in zip(buyers_no_advice, sellers_no_advice):
        state_buyer = buyer.get_state()
        state_seller = seller.get_state()

        # ε-greedy action selection
        if random.random() > epsilon:
            with torch.no_grad():
                action_buyer = q_network_no_advice(state_buyer).argmax().item()
                action_seller = q_network_no_advice(state_seller).argmax().item()
        else:
            action_buyer = random.choice(range(action_size))
            action_seller = random.choice(range(action_size))

        # RL-based price update
        buyer.update_bid_rl(action_buyer)
        seller.update_price_rl(action_seller)

        # Determine if a transaction occurs
        if buyer.bid_price >= seller.ask_price:
            raw_profit = max(0.0, buyer.utility - seller.cost)
            raw_utility = max(0.0, buyer.utility - buyer.bid_price)

            profit_buyer = raw_utility * buyer.beta
            profit_seller = raw_profit * seller.beta

            transactions_no_advice += 1
            round_profit_no_advice += profit_seller
            round_utility_no_advice += profit_buyer

            feedback_buyer = "success"
            feedback_seller = "success"
        else:
            profit_buyer = -1.0 * buyer.beta
            profit_seller = -1.0 * seller.beta
            feedback_buyer = "failure"
            feedback_seller = "failure"

        # Store prices for metrics
        bid_prices_round_no_advice.append(buyer.bid_price)
        ask_prices_round_no_advice.append(seller.ask_price)

        # Q-learning update every 10 rounds, using each agent's beta
        if round_num % 10 == 0:
            next_state_buyer = buyer.get_state()
            next_state_seller = seller.get_state()
            with torch.no_grad():
                target_buyer = profit_buyer + buyer.beta * q_network_no_advice(next_state_buyer).max().item()
                target_seller = profit_seller + seller.beta * q_network_no_advice(next_state_seller).max().item()

            pred_buyer = q_network_no_advice(state_buyer)[action_buyer]
            pred_seller = q_network_no_advice(state_seller)[action_seller]
            loss_buyer = F.mse_loss(pred_buyer, torch.tensor(target_buyer, dtype=torch.float32))
            loss_seller = F.mse_loss(pred_seller, torch.tensor(target_seller, dtype=torch.float32))

            optimizer_no_advice.zero_grad()
            (loss_buyer + loss_seller).backward()
            optimizer_no_advice.step()

    # After all pairings, imitation among peers (no-advice group)
    for group_agents, is_buyer_group in [(buyers_no_advice, True), (sellers_no_advice, False)]:
        for agent in group_agents:
            if random.random() < imitation_probability:
                peer = random.choice([a for a in group_agents if a.index != agent.index])
                if is_buyer_group:
                    agent.imitate_peer(peer.bid_price)
                else:
                    agent.imitate_peer(peer.ask_price)

    # Record metrics for the "no-advice" simulation
    transactions_per_round_no_advice.append(transactions_no_advice)
    profits_over_time_no_advice.append(round_profit_no_advice)
    utilities_over_time_no_advice.append(round_utility_no_advice)
    overall_bid_prices_no_advice.append(np.mean(bid_prices_round_no_advice) if bid_prices_round_no_advice else 0.0)
    overall_ask_prices_no_advice.append(np.mean(ask_prices_round_no_advice) if ask_prices_round_no_advice else 0.0)
    bid_ask_spread_no_advice.append(
        np.mean(np.array(ask_prices_round_no_advice) - np.array(bid_prices_round_no_advice))
        if bid_prices_round_no_advice else 0.0
    )

    # -----------------------------
    # Decay epsilon
    # -----------------------------
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    epsilon_values.append(epsilon)

    # Optional: print progress every 100 rounds
    if (round_num + 1) % 100 == 0 or (round_num == num_rounds - 1):
        print(f"Round {round_num + 1:3d} (With LLM): Tx = {transactions_advice:2d}, (No LLM): Tx = {transactions_no_advice:2d}")

print("Simulazione completata!")

# -----------------------------
# Plotting results
# -----------------------------
plt.figure(figsize=(18, 15))

# 1) Transactions per round
plt.subplot(3, 2, 1)
plt.plot(transactions_per_round_advice, label="Con LLM + Imitazione", linestyle='-', linewidth=1)
plt.plot(transactions_per_round_no_advice, label="Senza LLM + Imitazione", linestyle='--', linewidth=1)
plt.xlabel("Round")
plt.ylabel("Numero di transazioni")
plt.title("Transazioni per Round")
plt.legend()
plt.grid(True)

# 2) Profitti nel tempo
plt.subplot(3, 2, 2)
plt.plot(profits_over_time_advice, label="Con LLM + Imitazione", linestyle='-', linewidth=1)
plt.plot(profits_over_time_no_advice, label="Senza LLM + Imitazione", linestyle='--', linewidth=1)
plt.xlabel("Round")
plt.ylabel("Profitto totale")
plt.title("Profitti nel Tempo")
plt.legend()
plt.grid(True)

# 3) Utilità nel tempo
plt.subplot(3, 2, 3)
plt.plot(utilities_over_time_advice, label="Con LLM + Imitazione", linestyle='-', linewidth=1)
plt.plot(utilities_over_time_no_advice, label="Senza LLM + Imitazione", linestyle='--', linewidth=1)
plt.xlabel("Round")
plt.ylabel("Utilità totale")
plt.title("Utilità nel Tempo")
plt.legend()
plt.grid(True)

# 4) Prezzo medio bid/ask
plt.subplot(3, 2, 4)
plt.plot(overall_bid_prices_advice, label="Bid Medio (Con LLM)", linestyle='--', linewidth=1)
plt.plot(overall_ask_prices_advice, label="Ask Medio (Con LLM)", linestyle='-', linewidth=1)
plt.plot(overall_bid_prices_no_advice, label="Bid Medio (Senza LLM)", linestyle='--', linewidth=1, color='red')
plt.plot(overall_ask_prices_no_advice, label="Ask Medio (Senza LLM)", linestyle='-', linewidth=1, color='red')
plt.xlabel("Round")
plt.ylabel("Prezzo")
plt.title("Prezzo Medio Bid/Ask nel Tempo")
plt.legend()
plt.grid(True)

# 5) Spread bid-ask
plt.subplot(3, 2, 5)
plt.plot(bid_ask_spread_advice, label="Spread (Con LLM)", linestyle='-', linewidth=1)
plt.plot(bid_ask_spread_no_advice, label="Spread (Senza LLM)", linestyle='--', linewidth=1)
plt.xlabel("Round")
plt.ylabel("Spread")
plt.title("Spread Bid-Ask nel Tempo")
plt.legend()
plt.grid(True)

# 6) Epsilon decay
plt.subplot(3, 2, 6)
plt.plot(epsilon_values, label="Valore di Epsilon", linestyle='-', linewidth=1, color='green')
plt.xlabel("Round")
plt.ylabel("Epsilon")
plt.title("Decadimento di Epsilon")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
