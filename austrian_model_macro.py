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
from dataclasses import dataclass
from typing import List, Dict
from enum import Enum

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
# Austrian School Classes
# -----------------------------
class ProductionStage(Enum):
    RAW_MATERIALS = 1      # Beni di ordine superiore
    INTERMEDIATE = 2       # Beni intermedi
    CONSUMER_GOODS = 3     # Beni di consumo

@dataclass
class CapitalGood:
    stage: ProductionStage
    time_to_completion: int  # Roundaboutness del processo produttivo
    productivity: float
    depreciation_rate: float = 0.05

class CentralBank:
    def __init__(self):
        self.money_supply = 10000
        self.base_interest_rate = 0.03
        self.intervention_policy = "none"  # "expansionary", "contractionary", "none"
        
    def monetary_intervention(self, round_num: int) -> float:
        """Simula interventi di politica monetaria"""
        if self.intervention_policy == "expansionary":
            # Credit expansion - tasso artificialmente basso
            if round_num < 150:
                return max(0.01, self.base_interest_rate - 0.02)
            else:
                # Bust: il mercato corregge il tasso
                return self.base_interest_rate + 0.03
        return self.base_interest_rate
    
    def increase_money_supply(self, percentage: float):
        """Espansione monetaria che distorce i segnali di prezzo"""
        old_supply = self.money_supply
        self.money_supply *= (1 + percentage)
        # Effetto Cantillon: chi riceve i soldi prima beneficia di più
        return self.money_supply - old_supply

class LoanablesFundsMarket:
    def __init__(self):
        self.natural_interest_rate = 0.04  # Determinato da preferenze temporali
        self.market_interest_rate = 0.04
        self.artificial_credit = 0.0
        
    def update_rates(self, savings: float, investment_demand: float, central_bank: CentralBank, round_num: int):
        """Aggiorna i tassi secondo la teoria austriaca"""
        # Tasso naturale determinato da risparmio/investimento reali
        if savings > 0:
            self.natural_interest_rate = investment_demand / savings * 0.04
        
        # Tasso di mercato può essere distorto dall'intervento della banca centrale
        self.market_interest_rate = central_bank.monetary_intervention(round_num)
        
        # Differenza = credito artificiale
        self.artificial_credit = max(0, self.natural_interest_rate - self.market_interest_rate)

class BusinessCycleTracker:
    def __init__(self):
        self.phase = "neutral"  # "boom", "bust", "neutral"
        self.malinvestment_index = 0.0
        self.capital_consumption = 0.0
        self.entrepreneurial_errors = []
        
    def update_cycle_phase(self, artificial_credit: float, failed_projects: int, round_num: int):
        """Identifica la fase del ciclo secondo la teoria austriaca"""
        
        # BOOM PHASE: Credito artificiale causa mal-investimenti
        if artificial_credit > 0.01:
            self.phase = "boom"
            self.malinvestment_index += artificial_credit * 10
            
        # BUST PHASE: Correzione del mercato
        elif self.malinvestment_index > 5.0 and failed_projects > 3:
            self.phase = "bust"
            self.capital_consumption += failed_projects * 0.5
            
        # RECOVERY: Liquidazione completata
        elif self.capital_consumption > 0 and failed_projects < 2:
            self.phase = "neutral"
            self.malinvestment_index *= 0.8  # Graduale correzione
            
    def get_cycle_effects(self) -> Dict[str, float]:
        """Restituisce gli effetti del ciclo sui prezzi e produzione"""
        if self.phase == "boom":
            return {
                'price_inflation': 1.05,
                'investment_bias': 1.3,  # Troppi investimenti di lungo termine
                'employment_distortion': 1.2
            }
        elif self.phase == "bust":
            return {
                'price_deflation': 0.95,
                'liquidation_pressure': 0.7,
                'unemployment_spike': 1.4
            }
        return {'neutral': 1.0}

class AustrianConsumer:
    def __init__(self, consumer_id: int):
        self.consumer_id = consumer_id
        self.time_preference = random.uniform(0.01, 0.10)  # Eterogenea!
        self.income = random.uniform(80, 120)
        self.savings_rate = 1 / (1 + self.time_preference)  # Più alta preferenza = meno risparmio
        self.consumption_basket = {}
        
    def consumption_decision(self, prices: Dict[str, float], interest_rate: float):
        """Decisione consumo/risparmio basata su preferenza temporale"""
        # Se interesse_rate > time_preference → più risparmio
        if interest_rate > self.time_preference:
            self.savings_rate = min(0.8, self.savings_rate * 1.1)
        else:
            self.savings_rate = max(0.1, self.savings_rate * 0.9)
        
        consumption_budget = self.income * (1 - self.savings_rate)
        savings = self.income * self.savings_rate
        
        return consumption_budget, savings

class AustrianEntrepreneur:
    def __init__(self, entrepreneur_id: int):
        self.entrepreneur_id = entrepreneur_id
        self.alertness = random.uniform(0.3, 1.0)  # Kirznerian alertness
        self.past_profits = []
        self.current_projects = []
        
    def spot_profit_opportunity(self, price_discrepancies: List[tuple]) -> bool:
        """Scopre opportunità di profitto grazie all'alertness"""
        for price_diff, location_a, location_b in price_discrepancies:
            if random.random() < self.alertness and price_diff > 0.1:
                return True
        return False
    
    def coordinate_production(self, interest_rate: float, input_prices: Dict, output_prices: Dict):
        """Coordina la produzione intertemporale"""
        # Calcola se il progetto è profittevole dato il tasso di interesse
        expected_profit = self.calculate_discounted_profit(input_prices, output_prices, interest_rate)
        
        if expected_profit > 0:
            self.current_projects.append({
                'start_round': 0,
                'duration': random.randint(3, 8),  # Roundaboutness
                'expected_profit': expected_profit
            })
    
    def calculate_discounted_profit(self, input_prices: Dict, output_prices: Dict, interest_rate: float) -> float:
        """Calcola il profitto scontato di un progetto"""
        # Semplificazione: profitto base random
        base_profit = random.uniform(10, 50)
        time_periods = random.randint(2, 6)
        return base_profit / ((1 + interest_rate) ** time_periods)

class AustrianFirm:
    def __init__(self, firm_id: int, stage: ProductionStage):
        self.firm_id = firm_id
        self.stage = stage
        self.capital_goods: List[CapitalGood] = []
        self.cash_holdings = random.uniform(100, 500)
        self.time_preference = random.uniform(0.02, 0.08)  # Preferenza temporale individuale
        self.production_plan = []
        
    def calculate_present_value(self, future_revenue: float, time_periods: int) -> float:
        """Calcola il valore presente usando la preferenza temporale individuale"""
        discount_rate = self.time_preference
        return future_revenue / ((1 + discount_rate) ** time_periods)
    
    def make_investment_decision(self, interest_rate: float) -> bool:
        """Investe solo se il rendimento atteso > tasso di interesse di mercato"""
        expected_return = self.calculate_expected_return()
        return expected_return > interest_rate
    
    def calculate_expected_return(self) -> float:
        """Calcola il rendimento atteso degli investimenti"""
        if not self.capital_goods:
            return random.uniform(0.02, 0.08)
        return sum(cg.productivity for cg in self.capital_goods) / len(self.capital_goods)

# -----------------------------
# Originali classi con modifiche austriache
# -----------------------------
class Buyer:
    def __init__(self, utility: float, index: int):
        self.utility = utility
        # Reservation price ora dipende dalla preferenza temporale
        self.time_preference = random.uniform(0.01, 0.08)
        self.reservation_price = random.uniform(3.0, 7.5) * (1 + self.time_preference)
        # Initial bid price
        self.bid_price = max(self.reservation_price, utility * random.uniform(0.6, 1.0))
        # Index to identify cluster membership
        self.index = index
        # Austrian additions
        self.savings = random.uniform(50, 200)
        self.consumption_preference = random.uniform(0.3, 0.9)

    def get_state(self) -> torch.Tensor:
        """Return normalized state tensor [utility, reservation_price, bid_price]."""
        raw_state = torch.tensor([
            self.utility,
            self.reservation_price,
            self.bid_price
        ], dtype=torch.float32)
        return self.normalize_state(raw_state)
    
    def normalize_state(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """Normalize a state tensor by its maximum element to keep values in [0,1]."""
        eps = 1e-6
        return state_tensor / (torch.max(state_tensor) + eps)

    def update_bid_rl(self, action: int):
        """Update bid_price according to Q-learning action."""
        if action == 0:  # Increase price
            self.bid_price *= random.uniform(1.05, 1.1)
        elif action == 1:  # Decrease price
            self.bid_price *= random.uniform(0.9, 0.95)
        # Ensure boundaries
        self.bid_price = max(self.bid_price, self.reservation_price)
        self.bid_price = min(self.bid_price, self.reservation_price * 2)

    def imitate_peer(self, peer_bid: float):
        """With some probability, adopt a peer's bid price."""
        self.bid_price = peer_bid
    
    def savings_decision(self, interest_rate: float) -> float:
        """Decide how much to save based on time preference and interest rate"""
        if interest_rate > self.time_preference:
            return self.savings * random.uniform(1.1, 1.2)
        else:
            return self.savings * random.uniform(0.9, 1.0)

class Seller:
    def __init__(self, cost: float, index: int):
        self.cost = cost
        # Reservation price ora dipende dalla preferenza temporale
        self.time_preference = random.uniform(0.01, 0.08)
        self.reservation_price = random.uniform(3.0, 7.5) * (1 + self.time_preference)
        # Initial ask price
        self.ask_price = max(self.reservation_price, cost * random.uniform(1.1, 1.5))
        # Index to identify cluster membership
        self.index = index
        # Austrian additions
        self.capital_investment = random.uniform(20, 100)
        self.production_time = random.randint(1, 5)  # Roundaboutness

    def get_state(self) -> torch.Tensor:
        """Return normalized state tensor [cost, reservation_price, ask_price]."""
        raw_state = torch.tensor([
            self.cost,
            self.reservation_price,
            self.ask_price
        ], dtype=torch.float32)
        return self.normalize_state(raw_state)
    
    def normalize_state(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """Normalize a state tensor by its maximum element to keep values in [0,1]."""
        eps = 1e-6
        return state_tensor / (torch.max(state_tensor) + eps)

    def update_price_rl(self, action: int):
        """Update ask_price according to Q-learning action."""
        if action == 0:  # Increase price
            self.ask_price *= random.uniform(1.05, 1.1)
        elif action == 1:  # Decrease price
            self.ask_price *= random.uniform(0.9, 0.95)
        # Ensure boundaries
        self.ask_price = max(self.ask_price, self.reservation_price)
        self.ask_price = min(self.ask_price, self.reservation_price * 2)

    def imitate_peer(self, peer_ask: float):
        """With some probability, adopt a peer's ask price."""
        self.ask_price = peer_ask
    
    def investment_decision(self, interest_rate: float) -> bool:
        """Decide whether to invest based on expected return vs interest rate"""
        expected_return = random.uniform(0.02, 0.10)
        return expected_return > interest_rate

# -----------------------------
# Q-Network definition (unchanged)
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
# LLM-based advice system (unchanged)
# -----------------------------
def get_cluster_index(agent_index: int, total_agents: int) -> int:
    """Determine cluster index (0 or 1) based on agent index."""
    midpoint = total_agents // 2
    return 0 if agent_index < midpoint else 1

advice_cache = {}

def llm_advice(role: str, feedback: str, reservation_price: float, cluster: int, round_num: int) -> float:
    """Generate or retrieve cached advice for a given (role, cluster, feedback)."""
    global advice_cache

    cache_key = (role, cluster, feedback)
    advice_frequency = 50
    if (round_num % advice_frequency == 0) or (cache_key not in advice_cache):
        prompt = (
            f"In cluster {cluster}, a {role} had a '{feedback}' transaction. "
            f"Their reservation price is {reservation_price:.2f} euros. "
            "What should they do to adjust their price in the next round?"
        )
        try:
            response = generator(
                prompt,
                max_new_tokens=50,
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
    if "increase" in advice_text:
        return random.uniform(1.1, 1.2)
    elif "decrease" in advice_text:
        return random.uniform(0.85, 0.95)
    else:
        return random.uniform(0.95, 1.05)

# -----------------------------
# Austrian Macro Simulation Function
# -----------------------------
def run_austrian_macro_simulation(num_rounds: int = 500):
    """Simulazione macro secondo principi austriaci"""
    
    # Inizializzazione componenti macro
    central_bank = CentralBank()
    loanable_funds = LoanablesFundsMarket()
    business_cycle = BusinessCycleTracker()
    
    # Agenti eterogenei
    consumers = [AustrianConsumer(i) for i in range(20)]
    entrepreneurs = [AustrianEntrepreneur(i) for i in range(10)]
    firms = [AustrianFirm(i, random.choice(list(ProductionStage))) for i in range(15)]
    
    # Metriche macro
    gdp_over_time = []
    interest_rate_natural = []
    interest_rate_market = []
    malinvestment_index = []
    cycle_phases = []
    total_savings = []
    total_investment = []
    
    for round_num in range(num_rounds):
        
        # 1) POLITICA MONETARIA (può essere distorsiva)
        if round_num == 50:  # Inizio espansione monetaria
            central_bank.intervention_policy = "expansionary"
            print(f"Round {round_num}: Inizio ESPANSIONE MONETARIA")
        elif round_num == 200:  # Fine espansione
            central_bank.intervention_policy = "contractionary"
            print(f"Round {round_num}: Inizio CONTRAZIONE MONETARIA")
            
        # 2) MERCATO DEI FONDI MUTUABILI
        round_savings = sum(c.income * c.savings_rate for c in consumers)
        round_investment_demand = sum(len(f.capital_goods) * 10 for f in firms)
        
        loanable_funds.update_rates(round_savings, round_investment_demand, central_bank, round_num)
        
        # 3) DECISIONI INDIVIDUALI BASATE SU PREFERENZE TEMPORALI
        for consumer in consumers:
            consumption, savings = consumer.consumption_decision({}, loanable_funds.market_interest_rate)
            
        # 4) DECISIONI IMPRENDITORIALI
        failed_projects = 0
        for entrepreneur in entrepreneurs:
            # Simula fallimento progetti durante correzione di mercato
            if (round_num > 200 and 
                loanable_funds.artificial_credit > 0.02 and 
                business_cycle.phase == "boom"):
                if random.random() < 0.3:  # 30% prob di fallimento
                    failed_projects += 1
        
        # 5) DECISIONI DELLE IMPRESE
        for firm in firms:
            if firm.make_investment_decision(loanable_funds.market_interest_rate):
                # Aggiungi capitale
                new_capital = CapitalGood(
                    stage=firm.stage,
                    time_to_completion=random.randint(2, 6),
                    productivity=random.uniform(0.05, 0.15)
                )
                firm.capital_goods.append(new_capital)
        
        # 6) AGGIORNAMENTO CICLO ECONOMICO
        business_cycle.update_cycle_phase(
            loanable_funds.artificial_credit, 
            failed_projects, 
            round_num
        )
        
        # 7) CALCOLO GDP (semplificato)
        cycle_effects = business_cycle.get_cycle_effects()
        base_gdp = round_savings + round_investment_demand
        adjusted_gdp = base_gdp * cycle_effects.get('neutral', 1.0)
        if 'price_inflation' in cycle_effects:
            adjusted_gdp *= cycle_effects['price_inflation']
        elif 'price_deflation' in cycle_effects:
            adjusted_gdp *= cycle_effects['price_deflation']
            
        # 8) REGISTRAZIONE METRICHE
        gdp_over_time.append(adjusted_gdp)
        interest_rate_natural.append(loanable_funds.natural_interest_rate)
        interest_rate_market.append(loanable_funds.market_interest_rate)
        malinvestment_index.append(business_cycle.malinvestment_index)
        cycle_phases.append(business_cycle.phase)
        total_savings.append(round_savings)
        total_investment.append(round_investment_demand)
        
        if round_num % 50 == 0:
            print(f"Round {round_num}: GDP={adjusted_gdp:.1f}, "
                  f"Cycle={business_cycle.phase}, "
                  f"Natural Rate={loanable_funds.natural_interest_rate:.3f}, "
                  f"Market Rate={loanable_funds.market_interest_rate:.3f}")
    
    return {
        'gdp': gdp_over_time,
        'natural_rate': interest_rate_natural,
        'market_rate': interest_rate_market,
        'malinvestment': malinvestment_index,
        'cycle_phases': cycle_phases,
        'savings': total_savings,
        'investment': total_investment,
        'business_cycle': business_cycle
    }

def plot_austrian_results(results: Dict):
    """Grafici specifici per analisi austriaca"""
    
    plt.figure(figsize=(18, 15))
    
    # 1) Divergenza tassi di interesse
    plt.subplot(3, 3, 1)
    plt.plot(results['natural_rate'], label="Tasso Naturale", linewidth=2, color='blue')
    plt.plot(results['market_rate'], label="Tasso di Mercato", linewidth=2, color='red')
    plt.fill_between(range(len(results['natural_rate'])), 
                     results['natural_rate'], results['market_rate'], 
                     alpha=0.3, label="Credito Artificiale", color='yellow')
    plt.title("Divergenza Tassi di Interesse (Teoria Austriaca)")
    plt.xlabel("Round")
    plt.ylabel("Tasso di Interesse")
    plt.legend()
    plt.grid(True)
    
    # 2) PIL e cicli
    plt.subplot(3, 3, 2)
    plt.plot(results['gdp'], linewidth=2, color='blue')
    plt.axvspan(50, 200, alpha=0.2, color='green', label='Boom Artificiale')
    plt.axvspan(200, 300, alpha=0.2, color='red', label='Bust/Correzione')
    plt.title("PIL e Fasi del Ciclo Economico")
    plt.xlabel("Round")
    plt.ylabel("PIL")
    plt.legend()
    plt.grid(True)
    
    # 3) Indice di mal-investimento
    plt.subplot(3, 3, 3)
    plt.plot(results['malinvestment'], linewidth=2, color='red')
    plt.title("Indice di Mal-Investimento")
    plt.xlabel("Round")
    plt.ylabel("Intensità Mal-Investimenti")
    plt.grid(True)
    
    # 4) Risparmio vs Investimento
    plt.subplot(3, 3, 4)
    plt.plot(results['savings'], label="Risparmio", linewidth=2, color='green')
    plt.plot(results['investment'], label="Domanda Investimenti", linewidth=2, color='orange')
    plt.title("Risparmio vs Domanda di Investimenti")
    plt.xlabel("Round")
    plt.ylabel("Ammontare")
    plt.legend()
    plt.grid(True)
    
    # 5) Fasi del ciclo nel tempo
    plt.subplot(3, 3, 5)
    phases_numeric = []
    for phase in results['cycle_phases']:
        if phase == "boom":
            phases_numeric.append(1)
        elif phase == "bust":
            phases_numeric.append(-1)
        else:
            phases_numeric.append(0)
    
    plt.plot(phases_numeric, linewidth=3, color='purple')
    plt.title("Fasi del Ciclo Economico")
    plt.xlabel("Round")
    plt.ylabel("Fase (1=Boom, 0=Neutrale, -1=Bust)")
    plt.grid(True)
    
    # 6) Differenza tassi (credito artificiale)
    plt.subplot(3, 3, 6)
    artificial_credit = [nat - mkt for nat, mkt in zip(results['natural_rate'], results['market_rate'])]
    plt.plot(artificial_credit, linewidth=2, color='red')
    plt.title("Credito Artificiale (Naturale - Mercato)")
    plt.xlabel("Round")
    plt.ylabel("Differenza Tassi")
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# -----------------------------
# Simulation parameters
# -----------------------------
num_buyers = 10
num_sellers = 10
num_rounds = 500

# Buyers and sellers preferences (utility and cost) - now with Austrian modifications
buyers_utility = np.random.uniform(100, 200, num_buyers)
sellers_cost = np.random.uniform(50, 150, num_sellers)

# Q-learning parameters
state_size = 3
action_size = 3
gamma = 0.9
learning_rate = 0.001

# Exploration (epsilon-greedy)
initial_epsilon = 0.5
epsilon_decay = 0.99
min_epsilon = 0.1

# Imitation learning probability
imitation_probability = 0.1

# -----------------------------
# Initialize agents with Austrian features
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
# RUN AUSTRIAN MACRO SIMULATION
# -----------------------------
print("=== SIMULAZIONE MACROECONOMICA AUSTRIACA ===")
austrian_results = run_austrian_macro_simulation(num_rounds)

# Plot Austrian results
plot_austrian_results(austrian_results)

# -----------------------------
# Metrics and logging for original simulation
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
# Original Simulation loop with Austrian modifications
# -----------------------------
print("\n=== SIMULAZIONE MICROECONOMICA CON/SENZA LLM ===")
epsilon = initial_epsilon

# Initialize Austrian macro components for micro simulation
central_bank = CentralBank()
loanable_funds = LoanablesFundsMarket()

for round_num in range(num_rounds):
    # Update interest rates for Austrian influence
    total_savings = sum(buyer.savings_decision(loanable_funds.market_interest_rate) for buyer in buyers_advice)
    total_investment = sum(seller.capital_investment for seller in sellers_advice)
    loanable_funds.update_rates(total_savings, total_investment, central_bank, round_num)
    
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
        # Austrian influence: adjust reservation prices based on time preference and interest rates
        buyer.reservation_price *= (1 + (loanable_funds.market_interest_rate - buyer.time_preference) * 0.1)
        seller.reservation_price *= (1 + (loanable_funds.market_interest_rate - seller.time_preference) * 0.1)
        
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

        # Apply RL-based price update
        buyer.update_bid_rl(action_buyer)
        seller.update_price_rl(action_seller)

        # Determine if a transaction occurs
        if buyer.bid_price >= seller.ask_price:
            feedback_buyer = "success"
            feedback_seller = "success"
            transactions_advice += 1
            profit = max(0.0, buyer.utility - seller.cost)
            utility = max(0.0, buyer.utility - buyer.bid_price)
            round_profit_advice += profit
            round_utility_advice += utility
        else:
            feedback_buyer = "failure"
            feedback_seller = "failure"
            profit = -1.0
            utility = -1.0

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

        # Store prices for metrics
        bid_prices_round_advice.append(buyer.bid_price)
        ask_prices_round_advice.append(seller.ask_price)

        # Q-learning update every 10 rounds
        if round_num % 10 == 0:
            next_state_buyer = buyer.get_state()
            next_state_seller = seller.get_state()
            with torch.no_grad():
                target_buyer = profit + gamma * q_network_advice(next_state_buyer).max().item()
                target_seller = profit + gamma * q_network_advice(next_state_seller).max().item()

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
        # Austrian influence: adjust reservation prices
        buyer.reservation_price *= (1 + (loanable_funds.market_interest_rate - buyer.time_preference) * 0.1)
        seller.reservation_price *= (1 + (loanable_funds.market_interest_rate - seller.time_preference) * 0.1)
        
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
            transactions_no_advice += 1
            profit = max(0.0, buyer.utility - seller.cost)
            utility = max(0.0, buyer.utility - buyer.bid_price)
            round_profit_no_advice += profit
            round_utility_no_advice += utility
        else:
            profit = -1.0
            utility = -1.0

        # Store prices for metrics
        bid_prices_round_no_advice.append(buyer.bid_price)
        ask_prices_round_no_advice.append(seller.ask_price)

        # Q-learning update every 10 rounds
        if round_num % 10 == 0:
            next_state_buyer = buyer.get_state()
            next_state_seller = seller.get_state()
            with torch.no_grad():
                target_buyer = profit + gamma * q_network_no_advice(next_state_buyer).max().item()
                target_seller = profit + gamma * q_network_no_advice(next_state_seller).max().item()

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

    # Decay epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    epsilon_values.append(epsilon)

    # Optional: print progress
    if (round_num + 1) % 100 == 0 or (round_num == num_rounds - 1):
        print(f"Round {round_num + 1:3d} (With LLM): Transactions = {transactions_advice:2d}, "
              f"(No LLM): Transactions = {transactions_no_advice:2d}, "
              f"Interest Rate = {loanable_funds.market_interest_rate:.3f}")

print("Simulazione completata!")

# -----------------------------
# Plotting results - Original + Austrian
# -----------------------------
plt.figure(figsize=(20, 18))

# Original plots
plt.subplot(4, 3, 1)
plt.plot(transactions_per_round_advice, label="Con LLM + Imitazione", linestyle='-', linewidth=1)
plt.plot(transactions_per_round_no_advice, label="Senza LLM + Imitazione", linestyle='--', linewidth=1)
plt.xlabel("Round")
plt.ylabel("Numero di transazioni")
plt.title("Transazioni per Round")
plt.legend()
plt.grid(True)

plt.subplot(4, 3, 2)
plt.plot(profits_over_time_advice, label="Con LLM + Imitazione", linestyle='-', linewidth=1)
plt.plot(profits_over_time_no_advice, label="Senza LLM + Imitazione", linestyle='--', linewidth=1)
plt.xlabel("Round")
plt.ylabel("Profitto totale")
plt.title("Profitti nel Tempo")
plt.legend()
plt.grid(True)

plt.subplot(4, 3, 3)
plt.plot(utilities_over_time_advice, label="Con LLM + Imitazione", linestyle='-', linewidth=1)
plt.plot(utilities_over_time_no_advice, label="Senza LLM + Imitazione", linestyle='--', linewidth=1)
plt.xlabel("Round")
plt.ylabel("Utilità totale")
plt.title("Utilità nel Tempo")
plt.legend()
plt.grid(True)

plt.subplot(4, 3, 4)
plt.plot(overall_bid_prices_advice, label="Bid Medio (Con LLM)", linestyle='--', linewidth=1)
plt.plot(overall_ask_prices_advice, label="Ask Medio (Con LLM)", linestyle='-', linewidth=1)
plt.plot(overall_bid_prices_no_advice, label="Bid Medio (Senza LLM)", linestyle='--', linewidth=1, color='red')
plt.plot(overall_ask_prices_no_advice, label="Ask Medio (Senza LLM)", linestyle='-', linewidth=1, color='red')
plt.xlabel("Round")
plt.ylabel("Prezzo")
plt.title("Prezzo Medio Bid/Ask nel Tempo")
plt.legend()
plt.grid(True)

plt.subplot(4, 3, 5)
plt.plot(bid_ask_spread_advice, label="Spread (Con LLM)", linestyle='-', linewidth=1)
plt.plot(bid_ask_spread_no_advice, label="Spread (Senza LLM)", linestyle='--', linewidth=1)
plt.xlabel("Round")
plt.ylabel("Spread")
plt.title("Spread Bid-Ask nel Tempo")
plt.legend()
plt.grid(True)

plt.subplot(4, 3, 6)
plt.plot(epsilon_values, label="Valore di Epsilon", linestyle='-', linewidth=1, color='green')
plt.xlabel("Round")
plt.ylabel("Epsilon")
plt.title("Decadimento di Epsilon")
plt.legend()
plt.grid(True)

# Austrian additions to the plot
plt.subplot(4, 3, 7)
plt.plot(austrian_results['natural_rate'], label="Tasso Naturale", linewidth=2, color='blue')
plt.plot(austrian_results['market_rate'], label="Tasso Mercato", linewidth=2, color='red')
plt.xlabel("Round")
plt.ylabel("Tasso")
plt.title("Tassi di Interesse Austriaci")
plt.legend()
plt.grid(True)

plt.subplot(4, 3, 8)
plt.plot(austrian_results['gdp'], linewidth=2, color='purple')
plt.xlabel("Round")
plt.ylabel("PIL")
plt.title("PIL Macroeconomico")
plt.grid(True)

plt.subplot(4, 3, 9)
plt.plot(austrian_results['malinvestment'], linewidth=2, color='red')
plt.xlabel("Round")
plt.ylabel("Mal-investimenti")
plt.title("Indice Mal-Investimenti")
plt.grid(True)

plt.subplot(4, 3, 10)
artificial_credit = [nat - mkt for nat, mkt in zip(austrian_results['natural_rate'], austrian_results['market_rate'])]
plt.plot(artificial_credit, linewidth=2, color='orange')
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.xlabel("Round")
plt.ylabel("Credito Artificiale")
plt.title("Distorsione Monetaria")
plt.grid(True)

plt.subplot(4, 3, 11)
time_preferences_buyers = [buyer.time_preference for buyer in buyers_advice]
time_preferences_sellers = [seller.time_preference for seller in sellers_advice]
plt.hist(time_preferences_buyers, alpha=0.5, label="Buyers", bins=10)
plt.hist(time_preferences_sellers, alpha=0.5, label="Sellers", bins=10)
plt.xlabel("Preferenza Temporale")
plt.ylabel("Frequenza")
plt.title("Distribuzione Preferenze Temporali")
plt.legend()
plt.grid(True)

plt.subplot(4, 3, 12)
phases_numeric = []
for phase in austrian_results['cycle_phases']:
    if phase == "boom":
        phases_numeric.append(1)
    elif phase == "bust":
        phases_numeric.append(-1)
    else:
        phases_numeric.append(0)

plt.plot(phases_numeric, linewidth=3, color='purple')
plt.xlabel("Round")
plt.ylabel("Fase Ciclo")
plt.title("Cicli Economici Austriaci")
plt.grid(True)

plt.tight_layout()
plt.show()

# Print final Austrian analysis
print("\n=== ANALISI FINALE AUSTRIACA ===")
print(f"Fase finale del ciclo: {austrian_results['business_cycle'].phase}")
print(f"Indice mal-investimenti finale: {austrian_results['malinvestment'][-1]:.2f}")
print(f"Tasso naturale finale: {austrian_results['natural_rate'][-1]:.3f}")
print(f"Tasso di mercato finale: {austrian_results['market_rate'][-1]:.3f}")
print(f"Credito artificiale finale: {artificial_credit[-1]:.3f}")

# Summary statistics
avg_time_pref_buyers = np.mean([buyer.time_preference for buyer in buyers_advice])
avg_time_pref_sellers = np.mean([seller.time_preference for seller in sellers_advice])
print(f"\nPreferenza temporale media buyers: {avg_time_pref_buyers:.3f}")
print(f"Preferenza temporale media sellers: {avg_time_pref_sellers:.3f}")

boom_rounds = sum(1 for phase in austrian_results['cycle_phases'] if phase == "boom")
bust_rounds = sum(1 for phase in austrian_results['cycle_phases'] if phase == "bust")
print(f"\nRound in fase BOOM: {boom_rounds}")
print(f"Round in fase BUST: {bust_rounds}")
print(f"Round in fase NEUTRALE: {num_rounds - boom_rounds - bust_rounds}")
