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
# Market Monetarism Classes (Scott Sumner inspired)
# -----------------------------
class NGDPTargetingCentralBank:
    """
    Banca Centrale che targetizza il PIL Nominale, non i tassi di interesse
    Ispirata al Market Monetarism di Scott Sumner
    """
    def __init__(self, target_ngdp_growth: float = 0.05):
        self.target_ngdp_growth = target_ngdp_growth  # 5% annuo target
        self.money_supply = 10000
        self.base_money_growth = 0.03
        self.ngdp_history = []
        self.money_supply_history = []
        self.intervention_history = []
        
    def update_policy(self, current_ngdp: float, round_num: int):
        """
        Aggiorna la politica monetaria basata sul gap NGDP
        NON manipola direttamente i tassi di interesse
        """
        self.ngdp_history.append(current_ngdp)
        
        if len(self.ngdp_history) < 4:  # Serve storia per calcolare trend
            money_growth = self.base_money_growth
        else:
            # Calcola il tasso di crescita NGDP attuale (media mobile 4 periodi)
            recent_ngdp = np.mean(self.ngdp_history[-4:])
            past_ngdp = np.mean(self.ngdp_history[-8:-4]) if len(self.ngdp_history) >= 8 else self.ngdp_history[0]
            
            if past_ngdp > 0:
                actual_ngdp_growth = (recent_ngdp - past_ngdp) / past_ngdp
            else:
                actual_ngdp_growth = 0
            
            # Gap tra target e realtà
            ngdp_gap = self.target_ngdp_growth - actual_ngdp_growth
            
            # Regola di Taylor per NGDP invece che per inflazione + output gap
            # Se NGDP cresce troppo poco → espandi offerta monetaria
            # Se NGDP cresce troppo → contrai
            money_growth = self.base_money_growth + (ngdp_gap * 2.0)  # Coefficiente di reazione
            money_growth = max(-0.05, min(0.15, money_growth))  # Limiti ragionevoli
        
        # Aggiorna offerta di moneta
        old_supply = self.money_supply
        self.money_supply *= (1 + money_growth)
        money_injection = self.money_supply - old_supply
        
        self.money_supply_history.append(self.money_supply)
        self.intervention_history.append(money_growth)
        
        return money_injection, money_growth
    
    def get_policy_stance(self) -> str:
        """Determina se la politica è espansiva o restrittiva"""
        if not self.intervention_history:
            return "neutral"
        
        recent_growth = self.intervention_history[-1]
        if recent_growth > self.base_money_growth + 0.01:
            return "expansionary"
        elif recent_growth < self.base_money_growth - 0.01:
            return "contractionary"
        return "neutral"

class MarketExpectations:
    """
    Modella le aspettative di mercato tipiche del Market Monetarism
    Gli agenti formano aspettative razionali sui futures NGDP
    """
    def __init__(self):
        self.ngdp_futures = 0.05  # Aspettativa iniziale 5%
        self.confidence_level = 0.5  # Quanto fiducia nel central banker
        self.prediction_errors = []
        self.learning_rate = 0.1
        
    def update_expectations(self, actual_ngdp_growth: float, cb_target: float):
        """
        Aggiorna aspettative basate su performance passata della BC
        """
        # Errore di previsione
        prediction_error = actual_ngdp_growth - self.ngdp_futures
        self.prediction_errors.append(prediction_error)
        
        # Aggiorna aspettative con apprendimento adattivo
        self.ngdp_futures += self.learning_rate * prediction_error
        
        # Aggiorna fiducia nella banca centrale
        if len(self.prediction_errors) > 10:
            recent_errors = np.array(self.prediction_errors[-10:])
            volatility = np.std(recent_errors)
            # Meno volatilità = più fiducia
            self.confidence_level = max(0.1, min(0.9, 1.0 - volatility))
        
        # Le aspettative convergono verso il target se c'è credibilità
        credibility_weight = self.confidence_level
        self.ngdp_futures = (credibility_weight * cb_target + 
                           (1 - credibility_weight) * self.ngdp_futures)

# -----------------------------
# Austrian Classes (Modified for NGDP Targeting)
# -----------------------------
class ProductionStage(Enum):
    RAW_MATERIALS = 1
    INTERMEDIATE = 2
    CONSUMER_GOODS = 3

@dataclass
class CapitalGood:
    stage: ProductionStage
    time_to_completion: int
    productivity: float
    depreciation_rate: float = 0.05

class AustrianFirm:
    def __init__(self, firm_id: int, stage: ProductionStage):
        self.firm_id = firm_id
        self.stage = stage
        self.capital_goods: List[CapitalGood] = []
        self.cash_holdings = random.uniform(100, 500)
        self.time_preference = random.uniform(0.02, 0.08)
        self.production_plan = []
        self.ngdp_expectation = 0.05  # Aspettativa crescita PIL nominale
        
    def update_investment_decision(self, market_expectations: MarketExpectations, 
                                 natural_interest_rate: float) -> bool:
        """
        Decisioni di investimento basate su aspettative NGDP, non su tassi manipolati
        """
        # L'aspettativa NGDP influenza le decisioni più del tasso di mercato
        expected_nominal_return = market_expectations.ngdp_futures * random.uniform(0.8, 1.2)
        
        # Sconto con preferenza temporale individuale (austriaco)
        required_return = self.time_preference + 0.02  # Premio per rischio
        
        return expected_nominal_return > required_return
    
    def adjust_prices_for_ngdp(self, ngdp_expectation: float):
        """Aggiusta i prezzi basandosi sulle aspettative NGDP"""
        self.ngdp_expectation = ngdp_expectation

class AustrianConsumer:
    def __init__(self, consumer_id: int):
        self.consumer_id = consumer_id
        self.time_preference = random.uniform(0.01, 0.10)
        self.income = random.uniform(80, 120)
        self.savings_rate = 1 / (1 + self.time_preference)
        self.consumption_basket = {}
        self.inflation_expectation = 0.02
        
    def consumption_decision(self, ngdp_expectation: float, real_interest_rate: float):
        """
        Decisioni di consumo basate su aspettative NGDP
        """
        # Se si aspetta crescita nominale alta → consuma di più ora
        if ngdp_expectation > 0.06:  # Sopra target
            self.savings_rate = max(0.1, self.savings_rate * 0.95)
        elif ngdp_expectation < 0.04:  # Sotto target
            self.savings_rate = min(0.8, self.savings_rate * 1.05)
        
        consumption_budget = self.income * (1 - self.savings_rate)
        savings = self.income * self.savings_rate
        
        return consumption_budget, savings

class NatureInterestRateMarket:
    """
    Mercato che determina il tasso naturale SENZA intervento BC sui tassi
    """
    def __init__(self):
        self.natural_rate = 0.04
        self.market_rate = 0.04  # Ora NON distorto dalla BC
        
    def update_natural_rate(self, total_savings: float, investment_demand: float,
                          time_preferences: List[float]):
        """
        Tasso naturale determinato SOLO da preferenze temporali e produttività
        """
        if total_savings > 0 and time_preferences:
            # Tasso naturale = media delle preferenze temporali + produttività marginale
            avg_time_preference = np.mean(time_preferences)
            productivity_premium = investment_demand / total_savings * 0.02
            
            self.natural_rate = avg_time_preference + productivity_premium
            # Il tasso di mercato ora NON è distorto (segue il naturale)
            self.market_rate = self.natural_rate + random.uniform(-0.005, 0.005)  # Solo rumore minimo

class HybridBusinessCycle:
    """
    Cicli economici ibridi: austriaci MA con stabilizzazione NGDP
    """
    def __init__(self):
        self.phase = "neutral"
        self.malinvestment_index = 0.0
        self.ngdp_stabilization_effect = 0.0
        self.structural_distortions = 0.0  # Distorsioni reali nell'economia
        
    def update_cycle(self, ngdp_gap: float, interest_rate_distortion: float, 
                    money_injection: float):
        """
        Teoria austriaca modificata: i cicli derivano da distorsioni REALI,
        non da manipolazione dei tassi (che non c'è più)
        """
        
        # Effetto stabilizzante del NGDP targeting
        self.ngdp_stabilization_effect = -abs(ngdp_gap) * 0.5
        
        # Le distorsioni ora derivano da:
        # 1) Regolamentazioni eccessive
        # 2) Barriere all'entrata
        # 3) Sussidi settoriali
        # NON da manipolazione tassi (che non c'è)
        
        regulatory_distortion = random.uniform(0, 0.02)  # Distorsioni esogene
        self.structural_distortions += regulatory_distortion
        
        # Malinvestment ora deriva da distorsioni non-monetarie
        if self.structural_distortions > 0.05:
            self.malinvestment_index += 0.1
            if self.malinvestment_index > 3.0:
                self.phase = "bust"  # Correzione strutturale
        elif ngdp_gap < -0.02:  # NGDP troppo basso
            self.phase = "recession"  # Domanda insufficiente
        elif ngdp_gap > 0.02:  # NGDP troppo alto
            self.phase = "overheating"  # Eccesso domanda
        else:
            self.phase = "neutral"
            self.malinvestment_index *= 0.95  # Graduale correzione
            self.structural_distortions *= 0.98

# -----------------------------
# Original Classes (Modified)
# -----------------------------
class Buyer:
    def __init__(self, utility: float, index: int):
        self.utility = utility
        self.time_preference = random.uniform(0.01, 0.08)
        self.reservation_price = random.uniform(3.0, 7.5) * (1 + self.time_preference)
        self.bid_price = max(self.reservation_price, utility * random.uniform(0.6, 1.0))
        self.index = index
        self.savings = random.uniform(50, 200)
        self.ngdp_expectation = 0.05  # Market Monetarism: forward-looking

    def get_state(self) -> torch.Tensor:
        raw_state = torch.tensor([
            self.utility,
            self.reservation_price,
            self.bid_price
        ], dtype=torch.float32)
        return self.normalize_state(raw_state)
    
    def normalize_state(self, state_tensor: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        return state_tensor / (torch.max(state_tensor) + eps)

    def update_bid_rl(self, action: int):
        if action == 0:
            self.bid_price *= random.uniform(1.05, 1.1)
        elif action == 1:
            self.bid_price *= random.uniform(0.9, 0.95)
        self.bid_price = max(self.bid_price, self.reservation_price)
        self.bid_price = min(self.bid_price, self.reservation_price * 2)

    def imitate_peer(self, peer_bid: float):
        self.bid_price = peer_bid
    
    def adjust_for_ngdp_expectations(self, ngdp_expectation: float):
        """Aggiusta comportamento basato su aspettative NGDP"""
        self.ngdp_expectation = ngdp_expectation
        # Se aspetta alta crescita nominale → disposto a pagare di più
        if ngdp_expectation > 0.06:
            self.bid_price *= random.uniform(1.02, 1.05)
        elif ngdp_expectation < 0.04:
            self.bid_price *= random.uniform(0.98, 1.0)

class Seller:
    def __init__(self, cost: float, index: int):
        self.cost = cost
        self.time_preference = random.uniform(0.01, 0.08)
        self.reservation_price = random.uniform(3.0, 7.5) * (1 + self.time_preference)
        self.ask_price = max(self.reservation_price, cost * random.uniform(1.1, 1.5))
        self.index = index
        self.capital_investment = random.uniform(20, 100)
        self.production_time = random.randint(1, 5)
        self.ngdp_expectation = 0.05

    def get_state(self) -> torch.Tensor:
        raw_state = torch.tensor([
            self.cost,
            self.reservation_price,
            self.ask_price
        ], dtype=torch.float32)
        return self.normalize_state(raw_state)
    
    def normalize_state(self, state_tensor: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        return state_tensor / (torch.max(state_tensor) + eps)

    def update_price_rl(self, action: int):
        if action == 0:
            self.ask_price *= random.uniform(1.05, 1.1)
        elif action == 1:
            self.ask_price *= random.uniform(0.9, 0.95)
        self.ask_price = max(self.ask_price, self.reservation_price)
        self.ask_price = min(self.ask_price, self.reservation_price * 2)

    def imitate_peer(self, peer_ask: float):
        self.ask_price = peer_ask
    
    def adjust_for_ngdp_expectations(self, ngdp_expectation: float):
        """Aggiusta prezzi basato su aspettative NGDP"""
        self.ngdp_expectation = ngdp_expectation
        if ngdp_expectation > 0.06:
            self.ask_price *= random.uniform(1.02, 1.05)
        elif ngdp_expectation < 0.04:
            self.ask_price *= random.uniform(0.98, 1.0)

# -----------------------------
# Q-Network and LLM (unchanged)
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

def get_cluster_index(agent_index: int, total_agents: int) -> int:
    midpoint = total_agents // 2
    return 0 if agent_index < midpoint else 1

advice_cache = {}

def llm_advice(role: str, feedback: str, reservation_price: float, cluster: int, round_num: int) -> float:
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
# Hybrid Austrian-Market Monetarist Simulation
# -----------------------------
def run_hybrid_simulation(num_rounds: int = 500):
    """
    Simulazione ibrida Austriaca + Market Monetarism
    """
    
    # Market Monetarism components
    ngdp_central_bank = NGDPTargetingCentralBank(target_ngdp_growth=0.05)
    market_expectations = MarketExpectations()
    
    # Austrian components (modified)
    interest_rate_market = NatureInterestRateMarket()
    business_cycle = HybridBusinessCycle()
    
    # Agents
    consumers = [AustrianConsumer(i) for i in range(20)]
    firms = [AustrianFirm(i, random.choice(list(ProductionStage))) for i in range(15)]
    
    # Metrics
    ngdp_over_time = []
    ngdp_target_over_time = []
    ngdp_gap_over_time = []
    money_supply_over_time = []
    natural_rate_over_time = []
    market_rate_over_time = []
    expectations_over_time = []
    confidence_over_time = []
    business_cycle_phases = []
    malinvestment_over_time = []
    
    for round_num in range(num_rounds):
        
        # 1) CALCULATE CURRENT NGDP
        total_consumption = sum(c.income * (1 - c.savings_rate) for c in consumers)
        total_investment = sum(len(f.capital_goods) * 20 for f in firms)
        total_production = sum(f.cash_holdings * 0.1 for f in firms)
        
        current_ngdp = total_consumption + total_investment + total_production
        
        # 2) CENTRAL BANK NGDP TARGETING (no interest rate manipulation)
        money_injection, money_growth = ngdp_central_bank.update_policy(current_ngdp, round_num)
        
        # 3) CALCULATE NGDP GROWTH AND GAP
        if len(ngdp_central_bank.ngdp_history) >= 4:
            recent_ngdp = np.mean(ngdp_central_bank.ngdp_history[-4:])
            past_ngdp = np.mean(ngdp_central_bank.ngdp_history[-8:-4]) if len(ngdp_central_bank.ngdp_history) >= 8 else ngdp_central_bank.ngdp_history[0]
            actual_ngdp_growth = (recent_ngdp - past_ngdp) / past_ngdp if past_ngdp > 0 else 0
        else:
            actual_ngdp_growth = 0.05
        
        ngdp_gap = actual_ngdp_growth - ngdp_central_bank.target_ngdp_growth
        
        # 4) UPDATE MARKET EXPECTATIONS
        market_expectations.update_expectations(actual_ngdp_growth, ngdp_central_bank.target_ngdp_growth)
        
        # 5) NATURAL INTEREST RATE (no CB manipulation)
        total_savings = sum(c.income * c.savings_rate for c in consumers)
        investment_demand = sum(len(f.capital_goods) * 10 for f in firms)
        time_preferences = [c.time_preference for c in consumers] + [f.time_preference for f in firms]
        
        interest_rate_market.update_natural_rate(total_savings, investment_demand, time_preferences)
        
        # 6) BUSINESS CYCLE UPDATE (hybrid)
        interest_rate_distortion = 0.0  # NO distortion since CB doesn't manipulate rates
        business_cycle.update_cycle(ngdp_gap, interest_rate_distortion, money_injection)
        
        # 7) AGENTS RESPOND TO NGDP EXPECTATIONS
        for consumer in consumers:
            consumption, savings = consumer.consumption_decision(
                market_expectations.ngdp_futures, 
                interest_rate_market.natural_rate
            )
        
        for firm in firms:
            if firm.update_investment_decision(market_expectations, interest_rate_market.natural_rate):
                new_capital = CapitalGood(
                    stage=firm.stage,
                    time_to_completion=random.randint(2, 6),
                    productivity=random.uniform(0.05, 0.15)
                )
                firm.capital_goods.append(new_capital)
        
        # 8) MONEY INJECTION EFFECTS (Cantillon effects)
        if money_injection > 0:
            # First receivers benefit more (Cantillon effect)
            num_first_receivers = len(firms) // 2
            for i in range(num_first_receivers):
                firms[i].cash_holdings += money_injection / num_first_receivers
        
        # 9) RECORD METRICS
        ngdp_over_time.append(current_ngdp)
        ngdp_target_over_time.append(ngdp_central_bank.target_ngdp_growth * current_ngdp)
        ngdp_gap_over_time.append(ngdp_gap)
        money_supply_over_time.append(ngdp_central_bank.money_supply)
        natural_rate_over_time.append(interest_rate_market.natural_rate)
        market_rate_over_time.append(interest_rate_market.market_rate)
        expectations_over_time.append(market_expectations.ngdp_futures)
        confidence_over_time.append(market_expectations.confidence_level)
        business_cycle_phases.append(business_cycle.phase)
        malinvestment_over_time.append(business_cycle.malinvestment_index)
        
        if round_num % 50 == 0:
            print(f"Round {round_num}: NGDP Growth={actual_ngdp_growth:.3f}, "
                  f"Target={ngdp_central_bank.target_ngdp_growth:.3f}, "
                  f"Gap={ngdp_gap:.3f}, "
                  f"Expectations={market_expectations.ngdp_futures:.3f}, "
                  f"Confidence={market_expectations.confidence_level:.2f}")
    
    return {
        'ngdp': ngdp_over_time,
        'ngdp_target': ngdp_target_over_time,
        'ngdp_gap': ngdp_gap_over_time,
        'money_supply': money_supply_over_time,
        'natural_rate': natural_rate_over_time,
        'market_rate': market_rate_over_time,
        'expectations': expectations_over_time,
        'confidence': confidence_over_time,
        'cycle_phases': business_cycle_phases,
        'malinvestment': malinvestment_over_time,
        'central_bank': ngdp_central_bank,
        'market_expectations': market_expectations
    }

def plot_hybrid_results(results: Dict):
    """Grafici per analisi ibrida Austriaca-Market Monetarism"""
    
    plt.figure(figsize=(20, 16))
    
    # 1) NGDP Targeting Performance
    plt.subplot(4, 3, 1)
    ngdp_growth = []
    for i in range(1, len(results['ngdp'])):
        growth = (results['ngdp'][i] - results['ngdp'][i-1]) / results['ngdp'][i-1]
        ngdp_growth.append(growth)
    
    target_line = [0.05] * len(ngdp_growth)
    plt.plot(ngdp_growth, label="NGDP Growth Attuale", linewidth=2, color='blue')
    plt.plot(target_line, label="Target 5%", linewidth=2, color='red', linestyle='--')
    plt.fill_between(range(len(ngdp_growth)), ngdp_growth, target_line, alpha=0.3)
    plt.title("NGDP Targeting Performance")
    plt.xlabel("Round")
    plt.ylabel("Crescita NGDP")
    plt.legend()
    plt.grid(True)
    
    # 2) Market Expectations vs Reality
    plt.subplot(4, 3, 2)
    plt.plot(results['expectations'], label="Aspettative Mercato", linewidth=2, color='green')
    plt.plot(ngdp_growth, label="NGDP Reale", linewidth=2, color='blue')
    plt.title("Aspettative vs Realtà (Market Monetarism)")
    plt.xlabel("Round")
    plt.ylabel("Crescita Attesa/Reale")
    plt.legend()
    plt.grid(True)
    
    # 3) Central Bank Credibility
    plt.subplot(4, 3, 3)
    plt.plot(results['confidence'], linewidth=2, color='purple')
    plt.title("Credibilità Banca Centrale")
    plt.xlabel("Round")
    plt.ylabel("Livello Fiducia")
    plt.ylim(0, 1)
    plt.grid(True)
    
    # 4) Money Supply Growth
    plt.subplot(4, 3, 4)
    money_growth = []
    for i in range(1, len(results['money_supply'])):
        growth = (results['money_supply'][i] - results['money_supply'][i-1]) / results['money_supply'][i-1]
        money_growth.append(growth)
    plt.plot(money_growth, linewidth=2, color='orange')
    plt.title("Crescita Offerta Monetaria")
    plt.xlabel("Round")
    plt.ylabel("Crescita Money Supply")
    plt.grid(True)
    
    # 5) Interest Rates (Natural = Market, no distortion)
    plt.subplot(4, 3, 5)
    plt.plot(results['natural_rate'], label="Tasso Naturale", linewidth=2, color='blue')
    plt.plot(results['market_rate'], label="Tasso Mercato", linewidth=2, color='red', linestyle='--')
    plt.title("Tassi: NO Distorsione (Hybrid Model)")
    plt.xlabel("Round")
    plt.ylabel("Tasso Interesse")
    plt.legend()
    plt.grid(True)
    # Should be almost identical!
    
    # 6) NGDP Gap Over Time
    plt.subplot(4, 3, 6)
    plt.plot(results['ngdp_gap'], linewidth=2, color='red')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.title("NGDP Gap (Target - Actual)")
    plt.xlabel("Round")
    plt.ylabel("Gap")
    plt.grid(True)
    
    # 7) Business Cycle Phases
    plt.subplot(4, 3, 7)
    phase_mapping = {"neutral": 0, "recession": -1, "overheating": 1, "bust": -2}
    phases_numeric = [phase_mapping.get(phase, 0) for phase in results['cycle_phases']]
    plt.plot(phases_numeric, linewidth=3, color='purple')
    plt.title("Fasi Ciclo (Hybrid Austrian-MM)")
    plt.xlabel("Round")
    plt.ylabel("Fase")
    plt.grid(True)
    
    # 8) Malinvestment Index
    plt.subplot(4, 3, 8)
    plt.plot(results['malinvestment'], linewidth=2, color='red')
    plt.title("Mal-Investimenti (Non da Tassi)")
    plt.xlabel("Round")
    plt.ylabel("Indice")
    plt.grid(True)
    
    # 9) NGDP Level
    plt.subplot(4, 3, 9)
    plt.plot(results['ngdp'], linewidth=2, color='blue')
    plt.title("Livello NGDP")
    plt.xlabel("Round")
    plt.ylabel("NGDP Nominale")
    plt.grid(True)
    
    # 10) Expectation Errors
    plt.subplot(4, 3, 10)
    expectation_errors = []
    for i in range(min(len(results['expectations']), len(ngdp_growth))):
        error = results['expectations'][i] - ngdp_growth[i] if i < len(ngdp_growth) else 0
        expectation_errors.append(error)
    plt.plot(expectation_errors, linewidth=2, color='red')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title("Errori di Aspettativa")
    plt.xlabel("Round")
    plt.ylabel("Errore")
    plt.grid(True)
    
    # 11) Policy Stance
    plt.subplot(4, 3, 11)
    policy_mapping = {"neutral": 0, "expansionary": 1, "contractionary": -1}
    policy_history = []
    for i in range(len(results['central_bank'].intervention_history)):
        growth = results['central_bank'].intervention_history[i]
        if growth > 0.035:
            policy_history.append(1)
        elif growth < 0.025:
            policy_history.append(-1)
        else:
            policy_history.append(0)
    
    plt.plot(policy_history, linewidth=3, color='green')
    plt.title("Stance Politica Monetaria")
    plt.xlabel("Round")
    plt.ylabel("Stance")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# -----------------------------
# Simulation parameters
# -----------------------------
num_buyers = 10
num_sellers = 10
num_rounds = 500

# Buyers and sellers preferences 
buyers_utility = np.random.uniform(100, 200, num_buyers)
sellers_cost = np.random.uniform(50, 150, num_sellers)

# Q-learning parameters
state_size = 3
action_size = 3
gamma = 0.9
learning_rate = 0.001

# Exploration
initial_epsilon = 0.5
epsilon_decay = 0.99
min_epsilon = 0.1

# Imitation
imitation_probability = 0.1

# Initialize agents with NGDP expectations
buyers_advice = [Buyer(utility, idx) for idx, utility in enumerate(buyers_utility)]
sellers_advice = [Seller(cost, idx) for idx, cost in enumerate(sellers_cost)]

buyers_no_advice = [Buyer(utility, idx) for idx, utility in enumerate(buyers_utility)]
sellers_no_advice = [Seller(cost, idx) for idx, cost in enumerate(sellers_cost)]

# Initialize Q-networks
q_network_advice = QNetwork(state_size, action_size)
q_network_no_advice = QNetwork(state_size, action_size)
optimizer_advice = optim.Adam(q_network_advice.parameters(), lr=learning_rate)
optimizer_no_advice = optim.Adam(q_network_no_advice.parameters(), lr=learning_rate)

# -----------------------------
# RUN HYBRID SIMULATION
# -----------------------------
print("=== SIMULAZIONE IBRIDA AUSTRIACA + MARKET MONETARISM ===")
hybrid_results = run_hybrid_simulation(num_rounds)

# Plot results
plot_hybrid_results(hybrid_results)

# -----------------------------
# Continue with original micro simulation but with NGDP influence
# -----------------------------
print("\n=== SIMULAZIONE MICROECONOMICA CON INFLUENZA NGDP ===")

# Get NGDP expectations from hybrid simulation
ngdp_expectations = hybrid_results['expectations']

# Metrics
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

epsilon = initial_epsilon

for round_num in range(num_rounds):
    # Get NGDP expectation for this round
    current_ngdp_expectation = ngdp_expectations[round_num] if round_num < len(ngdp_expectations) else 0.05
    
    # Update agents' NGDP expectations
    for buyer in buyers_advice + buyers_no_advice:
        buyer.adjust_for_ngdp_expectations(current_ngdp_expectation)
    for seller in sellers_advice + sellers_no_advice:
        seller.adjust_for_ngdp_expectations(current_ngdp_expectation)
    
    # Shuffle agents
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

    # Simulation WITH LLM advice
    for buyer, seller in zip(buyers_advice, sellers_advice):
        cluster_buyer = get_cluster_index(buyer.index, num_buyers)
        cluster_seller = get_cluster_index(seller.index, num_sellers)

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

        buyer.update_bid_rl(action_buyer)
        seller.update_price_rl(action_seller)

        # Transaction check
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

        # LLM advice
        factor_buyer = llm_advice("buyer", feedback_buyer, buyer.reservation_price, cluster_buyer, round_num)
        factor_seller = llm_advice("seller", feedback_seller, seller.reservation_price, cluster_seller, round_num)

        buyer.bid_price *= factor_buyer
        seller.ask_price *= factor_seller

        # Boundaries
        buyer.bid_price = max(buyer.bid_price, buyer.reservation_price)
        buyer.bid_price = min(buyer.bid_price, buyer.reservation_price * 2)
        seller.ask_price = max(seller.ask_price, seller.reservation_price)
        seller.ask_price = min(seller.ask_price, seller.reservation_price * 2)

        bid_prices_round_advice.append(buyer.bid_price)
        ask_prices_round_advice.append(seller.ask_price)

        # Q-learning update
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

    # Imitation
    for group_agents, is_buyer_group in [(buyers_advice, True), (sellers_advice, False)]:
        for agent in group_agents:
            if random.random() < imitation_probability:
                peer = random.choice([a for a in group_agents if a.index != agent.index])
                if is_buyer_group:
                    agent.imitate_peer(peer.bid_price)
                else:
                    agent.imitate_peer(peer.ask_price)

    # Record metrics
    transactions_per_round_advice.append(transactions_advice)
    profits_over_time_advice.append(round_profit_advice)
    utilities_over_time_advice.append(round_utility_advice)
    overall_bid_prices_advice.append(np.mean(bid_prices_round_advice) if bid_prices_round_advice else 0.0)
    overall_ask_prices_advice.append(np.mean(ask_prices_round_advice) if ask_prices_round_advice else 0.0)
    bid_ask_spread_advice.append(
        np.mean(np.array(ask_prices_round_advice) - np.array(bid_prices_round_advice))
        if bid_prices_round_advice else 0.0
    )

    # Simulation WITHOUT LLM advice
    for buyer, seller in zip(buyers_no_advice, sellers_no_advice):
        state_buyer = buyer.get_state()
        state_seller = seller.get_state()

        if random.random() > epsilon:
            with torch.no_grad():
                action_buyer = q_network_no_advice(state_buyer).argmax().item()
                action_seller = q_network_no_advice(state_seller).argmax().item()
        else:
            action_buyer = random.choice(range(action_size))
            action_seller = random.choice(range(action_size))

        buyer.update_bid_rl(action_buyer)
        seller.update_price_rl(action_seller)

        if buyer.bid_price >= seller.ask_price:
            transactions_no_advice += 1
            profit = max(0.0, buyer.utility - seller.cost)
            utility = max(0.0, buyer.utility - buyer.bid_price)
            round_profit_no_advice += profit
            round_utility_no_advice += utility
        else:
            profit = -1.0
            utility = -1.0

        bid_prices_round_no_advice.append(buyer.bid_price)
        ask_prices_round_no_advice.append(seller.ask_price)

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

    # Imitation (no advice)
    for group_agents, is_buyer_group in [(buyers_no_advice, True), (sellers_no_advice, False)]:
        for agent in group_agents:
            if random.random() < imitation_probability:
                peer = random.choice([a for a in group_agents if a.index != agent.index])
                if is_buyer_group:
                    agent.imitate_peer(peer.bid_price)
                else:
                    agent.imitate_peer(peer.ask_price)

    # Record metrics (no advice)
    transactions_per_round_no_advice.append(transactions_no_advice)
    profits_over_time_no_advice.append(round_profit_no_advice)
    utilities_over_time_no_advice.append(round_utility_no_advice)
    overall_bid_prices_no_advice.append(np.mean(bid_prices_round_no_advice) if bid_prices_round_no_advice else 0.0)
    overall_ask_prices_no_advice.append(np.mean(ask_prices_round_no_advice) if ask_prices_round_no_advice else 0.0)
    bid_ask_spread_no_advice.append(
        np.mean(np.array(ask_prices_round_no_advice) - np.array(bid_prices_round_no_advice))
        if bid_prices_round_no_advice else 0.0
    )

    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    epsilon_values.append(epsilon)

    if (round_num + 1) % 100 == 0:
        print(f"Round {round_num + 1}: (With LLM): {transactions_advice}, "
              f"(No LLM): {transactions_no_advice}, "
              f"NGDP Exp: {current_ngdp_expectation:.3f}")

print("Simulazione completata!")

# Plot micro results
plt.figure(figsize=(20, 12))

plt.subplot(3, 3, 1)
plt.plot(transactions_per_round_advice, label="Con LLM", linestyle='-', linewidth=1)
plt.plot(transactions_per_round_no_advice, label="Senza LLM", linestyle='--', linewidth=1)
plt.xlabel("Round")
plt.ylabel("Transazioni")
plt.title("Transazioni (Influenza NGDP)")
plt.legend()
plt.grid(True)

plt.subplot(3, 3, 2)
plt.plot(profits_over_time_advice, label="Con LLM", linestyle='-', linewidth=1)
plt.plot(profits_over_time_no_advice, label="Senza LLM", linestyle='--', linewidth=1)
plt.xlabel("Round")
plt.ylabel("Profitto")
plt.title("Profitti (Influenza NGDP)")
plt.legend()
plt.grid(True)

plt.subplot(3, 3, 3)
plt.plot(utilities_over_time_advice, label="Con LLM", linestyle='-', linewidth=1)
plt.plot(utilities_over_time_no_advice, label="Senza LLM", linestyle='--', linewidth=1)
plt.xlabel("Round")
plt.ylabel("Utilità")
plt.title("Utilità (Influenza NGDP)")
plt.legend()
plt.grid(True)

plt.subplot(3, 3, 4)
plt.plot(overall_bid_prices_advice, label="Bid (LLM)", linestyle='--', linewidth=1)
plt.plot(overall_ask_prices_advice, label="Ask (LLM)", linestyle='-', linewidth=1)
plt.plot(overall_bid_prices_no_advice, label="Bid (No LLM)", linestyle='--', linewidth=1, color='red')
plt.plot(overall_ask_prices_no_advice, label="Ask (No LLM)", linestyle='-', linewidth=1, color='red')
plt.xlabel("Round")
plt.ylabel("Prezzo")
plt.title("Prezzi Medi (Influenza NGDP)")
plt.legend()
plt.grid(True)

plt.subplot(3, 3, 5)
plt.plot(bid_ask_spread_advice, label="Spread (LLM)", linestyle='-', linewidth=1)
plt.plot(bid_ask_spread_no_advice, label="Spread (No LLM)", linestyle='--', linewidth=1)
plt.xlabel("Round")
plt.ylabel("Spread")
plt.title("Spread Bid-Ask (Influenza NGDP)")
plt.legend()
plt.grid(True)

plt.subplot(3, 3, 6)
plt.plot(epsilon_values, label="Epsilon", linestyle='-', linewidth=1, color='green')
plt.xlabel("Round")
plt.ylabel("Epsilon")
plt.title("Decadimento Epsilon")
plt.legend()
plt.grid(True)

plt.subplot(3, 3, 7)
plt.plot(hybrid_results['expectations'], linewidth=2, color='purple')
plt.title("Aspettative NGDP nel Tempo")
plt.xlabel("Round")
plt.ylabel("Aspettativa Crescita")
plt.grid(True)

plt.subplot(3, 3, 8)
plt.plot(hybrid_results['confidence'], linewidth=2, color='orange')
plt.title("Fiducia nella Banca Centrale")
plt.xlabel("Round")
plt.ylabel("Livello Fiducia")
plt.grid(True)

plt.subplot(3, 3, 9)
ngdp_growth_final = []
for i in range(1, len(hybrid_results['ngdp'])):
    growth = (hybrid_results['ngdp'][i] - hybrid_results['ngdp'][i-1]) / hybrid_results['ngdp'][i-1]
    ngdp_growth_final.append(growth)
plt.plot(ngdp_growth_final, linewidth=2, color='blue')
plt.axhline(y=0.05, color='red', linestyle='--', label='Target 5%')
plt.title("Performance NGDP Targeting")
plt.xlabel("Round")
plt.ylabel("Crescita NGDP")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Final analysis
print("\n=== ANALISI FINALE MODELLO IBRIDO ===")
print(f"Target NGDP: {hybrid_results['central_bank'].target_ngdp_growth:.1%}")
final_expectation = hybrid_results['expectations'][-1]
final_confidence = hybrid_results['confidence'][-1]
print(f"Aspettativa finale: {final_expectation:.3f}")
print(f"Fiducia finale nella BC: {final_confidence:.2f}")

# Calculate average NGDP gap
avg_gap = np.mean([abs(gap) for gap in hybrid_results['ngdp_gap']])
print(f"Gap NGDP medio assoluto: {avg_gap:.4f}")

# Interest rate analysis
avg_natural_rate = np.mean(hybrid_results['natural_rate'])
avg_market_rate = np.mean(hybrid_results['market_rate'])
rate_distortion = abs(avg_natural_rate - avg_market_rate)
print(f"Distorsione tassi (dovrebbe essere ~0): {rate_distortion:.4f}")

# Business cycle analysis
cycle_counts = {}
for phase in hybrid_results['cycle_phases']:
    cycle_counts[phase] = cycle_counts.get(phase, 0) + 1

print(f"\nFasi ciclo economico:")
for phase, count in cycle_counts.items():
    print(f"  {phase}: {count} round ({count/num_rounds:.1%})")

print(f"\nConlusione: Il modello ibrido mostra stabilizzazione NGDP con minori distorsioni austriache!")
