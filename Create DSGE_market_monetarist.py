import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import fsolve
import seaborn as sns

class HeterogeneousAgentModel:
    def __init__(self, T=200):
        """
        Modello con agenti eterogenei e preferenza temporale endogena
        
        Tipologie di agenti:
        - Risparmiatori (households): alta pazienza, forniscono fondi
        - Prestatori (banks): intermediari, gestiscono rischio
        - Imprese: bassa pazienza, domandano fondi per investimenti
        """
        self.T = T
        
        # Parametri strutturali base
        self.alpha = 0.3          # quota capitale nella produzione
        self.beta = 0.7           # elasticità output gap a spesa aggregata
        self.sigma_inv = 0.5      # elasticità sostituzione intertemporale
        self.psi = 1.5            # elasticità r* rispetto a produttività
        self.phi = 0.5            # reazione BC al gap tassi interesse
        
        # Parametri agenti eterogenei
        self.setup_agent_parameters()
        
        # Parametri aspettative adattive
        self.lambda_pi = 0.3      # peso aspettative inflazione
        self.lambda_n = 0.4       # peso aspettative spesa aggregata
        
        # Tassi di crescita strutturali
        self.g_A = 0.05          # crescita produttività
        self.g_k = 0.03           # crescita capitale
        self.g_l = 0.01           # crescita lavoro
        
        # Volatilità shocks
        self.sigma_A = 0.001      # volatilità shock produttività
        self.sigma_demand = 0  # volatilità shock domanda a zero per simulare delle autorità pubbliche che stabilizzano la spesa aggregata
        self.sigma_monetary = 0.001  # volatilità shock monetari
        self.sigma_financial = 0.001  # volatilità shock finanziari
        
        # Inizializzazione
        self.reset_simulation()
    
    def setup_agent_parameters(self):
        """Configura parametri per i diversi tipi di agenti"""
        # Preferenze temporali per tipo di agente
        self.rho_savers = 0.01      # risparmiatori: molto pazienti
        self.rho_banks = 0.015      # banche: mediamente pazienti
        self.rho_firms = 0.035      # imprese: impazienti
        
        # Pesi relativi nell'economia (somma = 1)
        self.weight_savers = 0.6    # 60% dell'economia
        self.weight_banks = 0.1     # 10% dell'economia
        self.weight_firms = 0.3     # 30% dell'economia
        
        # Elasticità specifiche per agente
        self.sigma_savers = 0.3     # bassa elasticità sostituzione
        self.sigma_banks = 0.8      # alta elasticità (arbitraggio)
        self.sigma_firms = 0.6      # media elasticità
        
        # Parametri di rischio
        self.risk_aversion_savers = 2.0
        self.risk_aversion_banks = 1.2
        self.risk_aversion_firms = 0.8
        
        # Capacità di credito e vincoli finanziari
        self.leverage_constraint_firms = 3.0  # rapporto debito/capitale max
        self.capital_adequacy_banks = 0.08    # coefficiente patrimoniale
    
    def reset_simulation(self):
        """Reset delle variabili per nuova simulazione"""
        # Variabili aggregate (ereditate dal modello base)
        self.A_growth = np.zeros(self.T)
        self.k_growth = np.zeros(self.T)
        self.l_growth = np.zeros(self.T)
        self.y_star = np.zeros(self.T)
        self.y_tilde = np.zeros(self.T)
        self.y_total = np.zeros(self.T)
        
        self.n = np.zeros(self.T)
        self.n_star = np.zeros(self.T)
        self.pi = np.zeros(self.T)
        self.pi_target = np.zeros(self.T)
        
        # Preferenza temporale endogena
        self.rho_aggregate = np.zeros(self.T)
        self.r_natural = np.zeros(self.T)
        self.r_market = np.zeros(self.T)
        
        # Variabili per agenti specifici
        self.consumption_savers = np.zeros(self.T)
        self.savings_savers = np.zeros(self.T)
        self.lending_banks = np.zeros(self.T)
        self.spread_banks = np.zeros(self.T)
        self.investment_firms = np.zeros(self.T)
        self.leverage_firms = np.zeros(self.T)
        
        # Domanda e offerta di fondi
        self.fund_supply = np.zeros(self.T)
        self.fund_demand = np.zeros(self.T)
        self.credit_conditions = np.zeros(self.T)  # tightness del mercato del credito
        
        # Aspettative
        self.pi_expected = np.zeros(self.T)
        self.n_expected = np.zeros(self.T)
        
        # Shocks
        self.eps_A = np.zeros(self.T)
        self.eps_demand = np.zeros(self.T)
        self.eps_monetary = np.zeros(self.T)
        self.eps_financial = np.zeros(self.T)  # shock al sistema finanziario
        
        # Learning
        self.shock_recognition = np.zeros(self.T)

        # Decomposizione inflazione
        self.pi_supply = np.zeros(self.T)      # inflazione da offerta
        self.pi_demand = np.zeros(self.T)      # inflazione da domanda  
        self.pi_financial = np.zeros(self.T)   # inflazione da condizioni finanziarie
        self.pi_core = np.zeros(self.T)        # inflazione core (escl. volatili)
    
    def generate_shocks(self, seed=42):
        """Genera shocks stocastici inclusi quelli finanziari"""
        np.random.seed(seed)
        
        # Shock di produttività (persistenti)
        self.eps_A[0] = np.random.normal(0, self.sigma_A)
        for t in range(1, self.T):
            self.eps_A[t] = 0.7 * self.eps_A[t-1] + np.random.normal(0, self.sigma_A)
        
        # Altri shocks
        self.eps_demand = np.random.normal(0, self.sigma_demand, self.T)
        self.eps_monetary = np.random.normal(0, self.sigma_monetary, self.T)
        
        # Shock finanziari (correlati con ciclo economico)
        for t in range(self.T):
            if t == 0:
                self.eps_financial[t] = np.random.normal(0, self.sigma_financial)
            else:
                # Shock finanziari più probabili durante recessioni
                recession_indicator = self.y_tilde[t-1] < -0.01 if t > 0 else 0
                volatility = self.sigma_financial * (1 + 2 * recession_indicator)
                self.eps_financial[t] = (0.3 * self.eps_financial[t-1] + 
                                       np.random.normal(0, volatility))
    
    def update_agent_behavior(self, t):
        """Aggiorna comportamento dei diversi agenti"""
        # Condizioni macroeconomiche
        expected_growth = self.y_star[t] if t == 0 else 0.5 * (self.y_star[t] + self.y_star[t-1])
        uncertainty = abs(self.eps_A[t]) + abs(self.eps_financial[t])
        
        # RISPARMIATORI
        # Funzione di utilità con preferenza per la smoothness del consumo
        if t == 0:
            self.consumption_savers[t] = 0.7  # consumo iniziale normalizzato
        else:
            # Consumo reagisce a reddito permanente e tassi reali
            real_rate = self.r_market[t-1] - self.pi_expected[t]
            income_growth = expected_growth
            self.consumption_savers[t] = (self.consumption_savers[t-1] * 
                                        (1 + income_growth - 0.5 * real_rate))
        
        # Risparmi dei households
        income_level = 1 + expected_growth  # reddito normalizzato
        self.savings_savers[t] = income_level - self.consumption_savers[t]
        
        # BANCHE
        # Spread bancario dipende da rischio e condizioni di liquidità
        base_spread = 0.015  # spread base
        risk_premium = uncertainty * self.risk_aversion_banks
        liquidity_premium = max(0, self.credit_conditions[t-1] * 0.01) if t > 0 else 0
        
        self.spread_banks[t] = base_spread + risk_premium + liquidity_premium
        
        # Capacità di prestito delle banche
        bank_capital = 1.0  # capitale bancario normalizzato
        self.lending_banks[t] = bank_capital / self.capital_adequacy_banks
        
        # IMPRESE
        # Investimenti dipendono da produttività attesa e costo del capitale
        productivity_boost = self.A_growth[t]
        cost_of_capital = self.r_market[t] + self.spread_banks[t]
        
        # Vincolo di leverage
        if t == 0:
            self.leverage_firms[t] = 1.5  # leverage iniziale
        else:
            # Leverage evolve con profittabilità e vincoli
            profitability = expected_growth + productivity_boost
            max_leverage = self.leverage_constraint_firms * (1 - uncertainty)
            self.leverage_firms[t] = min(max_leverage, 
                                       self.leverage_firms[t-1] * (1 + 0.5 * profitability))
        
        # Investimenti
        investment_productivity = productivity_boost + expected_growth
        financial_constraint = min(1.0, self.leverage_firms[t] / self.leverage_constraint_firms)
        
        self.investment_firms[t] = (investment_productivity * financial_constraint - 
                                  0.5 * cost_of_capital + self.eps_financial[t])
    
    def update_credit_market(self, t):
        """Aggiorna equilibrio del mercato del credito"""
        # Offerta di fondi (da risparmiatori via banche)
        self.fund_supply[t] = (self.savings_savers[t] * self.weight_savers * 
                              min(1.0, self.lending_banks[t]))
        
        # Domanda di fondi (da imprese)
        self.fund_demand[t] = self.investment_firms[t] * self.weight_firms
        
        # Condizioni creditizie (excess demand)
        excess_demand = self.fund_demand[t] - self.fund_supply[t]
        if t == 0:
            self.credit_conditions[t] = excess_demand
        else:
            # Smoothing delle condizioni creditizie
            self.credit_conditions[t] = (0.7 * self.credit_conditions[t-1] + 
                                       0.3 * excess_demand)
    
    def update_aggregate_time_preference(self, t):
        """Calcola preferenza temporale aggregata endogena"""
        # Pesi dinamici basati su condizioni economiche
        # In recessioni, le imprese pesano di più (hanno più urgenza)
        # In espansioni, i risparmiatori pesano di più
        
        economic_state = self.y_tilde[t-1] if t > 0 else 0
        credit_stress = max(0, self.credit_conditions[t])
        
        # Aggiustamento dei pesi
        stress_factor = credit_stress + abs(economic_state) * 0.5
        
        adj_weight_savers = self.weight_savers * (1 - 0.3 * stress_factor)
        adj_weight_firms = self.weight_firms * (1 + 0.5 * stress_factor)
        adj_weight_banks = self.weight_banks
        
        # Normalizzazione
        total_weight = adj_weight_savers + adj_weight_banks + adj_weight_firms
        adj_weight_savers /= total_weight
        adj_weight_banks /= total_weight
        adj_weight_firms /= total_weight
        
        # Preferenza temporale aggregata
        self.rho_aggregate[t] = (adj_weight_savers * self.rho_savers +
                            adj_weight_banks * self.rho_banks +
                            adj_weight_firms * self.rho_firms)
        
        # Aggiustamento per condizioni finanziarie (limitato per mantenere positività)
        financial_stress = abs(self.eps_financial[t]) * 2
        self.rho_aggregate[t] += financial_stress  # stress aumenta impazienza
        
        # VINCOLO DI POSITIVITÀ: assicuriamo che ρ sia sempre >= 0.005 (0.5%)
        self.rho_aggregate[t] = max(0.005, self.rho_aggregate[t])

    def update_natural_rate(self, t):
        """Calcola tasso naturale con preferenza temporale endogena"""
        expected_growth = self.y_star[t] if t == 0 else 0.5 * (self.y_star[t] + self.y_star[t-1])
        
        # Elasticità aggregata di sostituzione intertemporale
        sigma_aggregate = (self.weight_savers * self.sigma_savers +
                        self.weight_banks * self.sigma_banks +
                        self.weight_firms * self.sigma_firms)
        
        # Tasso naturale con componenti endogene
        r_nat_base = (self.rho_aggregate[t] + 
                    sigma_aggregate * expected_growth + 
                    self.psi * self.A_growth[t] +
                    0.5 * self.credit_conditions[t])  # premio per tensioni creditizie
        
        # VINCOLO DI POSITIVITÀ: assicuriamo che r* sia sempre >= 0.001 (0.1%)
        # Questo è fondamentale perché r* negativo creerebbe distorsioni nelle decisioni di investimento
        self.r_natural[t] = max(0.001, r_nat_base)
    
    def update_fundamentals(self, t):
        """Aggiorna variabili fondamentali (ereditate dal modello base)"""
        self.A_growth[t] = self.g_A + self.eps_A[t]
        self.k_growth[t] = self.g_k + 0.3 * self.eps_A[t]
        self.l_growth[t] = self.g_l
        
        self.y_star[t] = (self.A_growth[t] + 
                         self.alpha * self.k_growth[t] + 
                         (1 - self.alpha) * self.l_growth[t])
        
        self.pi_target[t] = -self.A_growth[t]
        self.n_star[t] = self.alpha * self.k_growth[t] + (1 - self.alpha) * self.l_growth[t]
    
    def update_expectations(self, t):
        """Aggiorna aspettative (ereditate dal modello base)"""
        if t == 0:
            self.pi_expected[t] = 0
            self.n_expected[t] = self.n_star[t]
            self.shock_recognition[t] = 0
        else:
            self.pi_expected[t] = (self.lambda_pi * self.pi[t-1] + 
                                 (1 - self.lambda_pi) * self.pi_expected[t-1])
            
            self.n_expected[t] = (self.lambda_n * self.n[t-1] + 
                                (1 - self.lambda_n) * self.n_expected[t-1])
            
            # Learning process
            n_stability = abs(self.n[t-1] - self.n_star[t-1]) < 0.005
            deflation_persistent = np.mean(self.pi[max(0, t-5):t]) < -0.005
            
            if n_stability and deflation_persistent and t > 10:
                self.shock_recognition[t] = min(1.0, self.shock_recognition[t-1] + 0.1)
            else:
                self.shock_recognition[t] = max(0.0, self.shock_recognition[t-1] - 0.05)
    
    def central_bank_reaction(self, t):
        """Reazione della banca centrale - solo targeting della spesa nominale"""
        if t == 0:
            # Inizializzazione: spesa nominale al target più shock monetario
            self.n[t] = self.n_star[t] + self.eps_monetary[t]
            # Tasso di mercato segue passivamente il tasso naturale
            self.r_market[t] = self.r_natural[t]
        else:
            # REGOLA DI POLITICA MONETARIA - Solo spesa nominale
            
            # 1. Gap di spesa nominale
            n_gap = self.n_expected[t] - self.n_star[t]
            
            # 2. Preoccupazioni per stabilità finanziaria
            financial_stability_concern = abs(self.credit_conditions[t]) * 0.3
            
            # 3. Aggiustamento della spesa nominale
            n_adjustment = (-self.phi * n_gap -  # contrasta deviazioni dal target
                        financial_stability_concern +  # riduce spesa se tensioni creditizie
                        self.eps_monetary[t])  # shock di politica monetaria
            
            # 4. Learning e commitment sulla spesa nominale
            commitment_strength = 0.8 + 0.2 * self.shock_recognition[t]
            
            # 5. Regola finale per spesa nominale
            self.n[t] = (commitment_strength * self.n_star[t] + 
                        (1 - commitment_strength) * (self.n_star[t] + n_adjustment))
            
            # 6. Tasso di mercato segue passivamente (nessun targeting attivo)
            # La BC non controlla direttamente i tassi, che si adeguano al mercato
            r_gap = self.r_natural[t] - self.r_market[t-1]
            self.r_market[t] = self.r_market[t-1] + 0.6 * r_gap  # convergenza più lenta    

    def solve_equilibrium(self, t):
        """Risolve equilibrio del periodo"""
        n_gap = self.n[t] - self.n_expected[t]
        self.y_tilde[t] = self.beta * n_gap + self.eps_demand[t]
        
        self.y_total[t] = self.y_star[t] + self.y_tilde[t]
        self.pi[t] = self.n[t] - self.y_total[t]


    def decompose_inflation(self, t):
        """Decompone inflazione in componenti di offerta e domanda"""
        
        # Componente di offerta (produttività e capacità)
        supply_component = -self.eps_A[t]  # shock produttività negativi → inflazione
        capacity_component = -self.y_star[t] if t > 0 else 0
        
        # Componente di domanda (spesa aggregata e gap)
        demand_component = self.n[t] - self.n_star[t]  # eccesso di spesa nominale
        output_gap_component = -self.y_tilde[t]  # output gap negativo → inflazione
        
        # Componente finanziaria (condizioni creditizie)
        financial_component = self.credit_conditions[t] * 0.1  # tensioni → prezzi
        
        # Inflazione totale decomposta
        self.pi_supply[t] = supply_component + capacity_component
        self.pi_demand[t] = demand_component + output_gap_component
        self.pi_financial[t] = financial_component
        
        # Verifica coerenza
        total_decomposed = self.pi_supply[t] + self.pi_demand[t] + self.pi_financial[t]
        
        return {
            'supply': self.pi_supply[t],
            'demand': self.pi_demand[t], 
            'financial': self.pi_financial[t],
            'total_check': total_decomposed
    }
    
    def simulate(self, seed=42):
        """Esegue simulazione completa"""
        self.reset_simulation()
        self.generate_shocks(seed)
        
        for t in range(self.T):
            self.update_fundamentals(t)
            self.update_agent_behavior(t)
            self.update_credit_market(t)
            self.update_aggregate_time_preference(t)
            self.update_natural_rate(t)
            self.update_expectations(t)
            self.central_bank_reaction(t)
            self.solve_equilibrium(t)
            self.decompose_inflation(t)
        
        return self.create_results_dataframe()
    
    def create_results_dataframe(self):
        """Crea DataFrame con risultati estesi"""
        results = pd.DataFrame({
            'periodo': range(self.T),
            'crescita_produttivita': self.A_growth,
            'pil_potenziale': self.y_star,
            'output_gap': self.y_tilde,
            'pil_totale': self.y_total,
            'inflazione': self.pi,
            'inflazione_target': self.pi_target,
            'inflazione_attesa': self.pi_expected,
            'spesa_aggregata': self.n,
            'spesa_target': self.n_star,
            'tasso_naturale': self.r_natural,
            'tasso_mercato': self.r_market,
            'rho_aggregato': self.rho_aggregate,
            'consumo_risparmiatori': self.consumption_savers,
            'risparmi_risparmiatori': self.savings_savers,
            'prestiti_banche': self.lending_banks,
            'spread_bancario': self.spread_banks,
            'investimenti_imprese': self.investment_firms,
            'leverage_imprese': self.leverage_firms,
            'offerta_fondi': self.fund_supply,
            'domanda_fondi': self.fund_demand,
            'condizioni_creditizie': self.credit_conditions,
            'riconoscimento_shock': self.shock_recognition,
            # AGGIUNGI QUESTE LINEE:
            'pi_supply': self.pi_supply,
            'pi_demand': self.pi_demand,
            'pi_financial': self.pi_financial
        })
        
        # Conversione in percentuali
        for col in ['crescita_produttivita', 'pil_potenziale', 'output_gap', 'pil_totale', 
                   'inflazione', 'inflazione_target', 'inflazione_attesa', 
                   'spesa_aggregata', 'spesa_target', 'rho_aggregato',
                   'pi_supply', 'pi_demand', 'pi_financial']:  # AGGIUNGI QUESTE
            results[col] *= 100
        
        for col in ['tasso_naturale', 'tasso_mercato', 'spread_bancario']:
            results[col] *= 100
            
        return results
    
    def plot_heterogeneous_results(self, results):
        """Crea grafici specifici per il modello con agenti eterogenei"""
        fig, axes = plt.subplots(4, 2, figsize=(16, 20))
        
        # 1. Preferenza temporale endogena
        axes[0,0].plot(results['periodo'], results['rho_aggregato'], 'purple', linewidth=2)
        axes[0,0].axhline(y=self.rho_savers*100, color='blue', linestyle='--', alpha=0.7, label='Risparmiatori')
        axes[0,0].axhline(y=self.rho_banks*100, color='green', linestyle='--', alpha=0.7, label='Banche')
        axes[0,0].axhline(y=self.rho_firms*100, color='red', linestyle='--', alpha=0.7, label='Imprese')
        axes[0,0].set_title('Preferenza Temporale Aggregata Endogena')
        axes[0,0].set_ylabel('ρ aggregato (%)')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Mercato del credito
        axes[0,1].plot(results['periodo'], results['offerta_fondi'], 'blue', label='Offerta fondi')
        axes[0,1].plot(results['periodo'], results['domanda_fondi'], 'red', label='Domanda fondi')
        axes[0,1].fill_between(results['periodo'], 
                              results['offerta_fondi'], results['domanda_fondi'],
                              alpha=0.3, label='Squilibrio')
        axes[0,1].set_title('Mercato del Credito')
        axes[0,1].set_ylabel('Flussi di fondi')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Comportamento risparmiatori
        ax_twin = axes[1,0].twinx()
        axes[1,0].plot(results['periodo'], results['consumo_risparmiatori'], 'blue', label='Consumo')
        ax_twin.plot(results['periodo'], results['risparmi_risparmiatori'], 'green', label='Risparmi')
        axes[1,0].set_title('Comportamento Risparmiatori')
        axes[1,0].set_ylabel('Consumo', color='blue')
        ax_twin.set_ylabel('Risparmi', color='green')
        axes[1,0].legend(loc='upper left')
        ax_twin.legend(loc='upper right')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Settore bancario
        ax_twin2 = axes[1,1].twinx()
        axes[1,1].plot(results['periodo'], results['spread_bancario'], 'orange', label='Spread')
        ax_twin2.plot(results['periodo'], results['prestiti_banche'], 'purple', label='Prestiti')
        axes[1,1].set_title('Settore Bancario')
        axes[1,1].set_ylabel('Spread bancario (%)', color='orange')
        ax_twin2.set_ylabel('Capacità prestiti', color='purple')
        axes[1,1].legend(loc='upper left')
        ax_twin2.legend(loc='upper right')
        axes[1,1].grid(True, alpha=0.3)
        
        # 5. Settore imprese
        ax_twin3 = axes[2,0].twinx()
        axes[2,0].plot(results['periodo'], results['investimenti_imprese'], 'red', label='Investimenti')
        ax_twin3.plot(results['periodo'], results['leverage_imprese'], 'brown', label='Leverage')
        axes[2,0].set_title('Settore Imprese')
        axes[2,0].set_ylabel('Investimenti', color='red')
        ax_twin3.set_ylabel('Leverage', color='brown')
        axes[2,0].legend(loc='upper left')
        ax_twin3.legend(loc='upper right')
        axes[2,0].grid(True, alpha=0.3)
        
        # 6. Condizioni creditizie vs tasso naturale
        ax_twin4 = axes[2,1].twinx()
        axes[2,1].plot(results['periodo'], results['condizioni_creditizie'], 'red', label='Tensioni creditizie')
        ax_twin4.plot(results['periodo'], results['tasso_naturale'], 'blue', label='Tasso naturale')
        axes[2,1].set_title('Condizioni Finanziarie')
        axes[2,1].set_ylabel('Tensioni creditizie', color='red')
        ax_twin4.set_ylabel('Tasso naturale (%)', color='blue')
        axes[2,1].legend(loc='upper left')
        ax_twin4.legend(loc='upper right')
        axes[2,1].grid(True, alpha=0.3)
        
        # 7. Inflazione nel modello eterogeneo
        axes[3,0].plot(results['periodo'], results['inflazione'], 'b-', label='Inflazione effettiva')
        axes[3,0].plot(results['periodo'], results['inflazione_target'], 'r--', label='Target')
        axes[3,0].fill_between(results['periodo'], results['inflazione'], results['inflazione_target'],
                              alpha=0.3, label='Deviazione dal target')
        axes[3,0].set_title('Dinamiche Inflazionistiche')
        axes[3,0].set_ylabel('Inflazione (%)')
        axes[3,0].legend()
        axes[3,0].grid(True, alpha=0.3)
        
        # 8. Correlazione ρ aggregato - condizioni economiche
        scatter = axes[3,1].scatter(results['rho_aggregato'], results['output_gap'], 
                                  c=results['condizioni_creditizie'], cmap='coolwarm', alpha=0.6)
        axes[3,1].set_xlabel('ρ aggregato (%)')
        axes[3,1].set_ylabel('Output gap (%)')
        axes[3,1].set_title('Preferenza Temporale vs Ciclo Economico')
        plt.colorbar(scatter, ax=axes[3,1], label='Tensioni creditizie')
        axes[3,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Esempio di utilizzo
if __name__ == "__main__":
    # Crea e simula il modello con agenti eterogenei
    hetero_model = HeterogeneousAgentModel(T=150)
    hetero_results = hetero_model.simulate(seed=42)
    
    # Statistiche comparative
    print("=== MODELLO CON AGENTI ETEROGENEI ===")
    print(f"ρ aggregato medio: {hetero_results['rho_aggregato'].mean():.2f}%")
    print(f"ρ aggregato std: {hetero_results['rho_aggregato'].std():.2f}%")
    print(f"Tasso naturale medio: {hetero_results['tasso_naturale'].mean():.2f}%")
    print(f"Spread bancario medio: {hetero_results['spread_bancario'].mean():.2f}%")
    print(f"Tensioni creditizie medie: {hetero_results['condizioni_creditizie'].mean():.3f}")
    print(f"Correlazione ρ-output gap: {hetero_results['rho_aggregato'].corr(hetero_results['output_gap']):.3f}")
    #mostra altre statistiche per avere più dettagli
    print(f"Correlazione ρ-inflazione: {hetero_results['rho_aggregato'].corr(hetero_results['inflazione']):.3f}")
    print(f"Correlazione ρ-tasso naturale: {hetero_results['rho_aggregato'].corr(hetero_results['tasso_naturale']):.3f}")
    # mostra statistiche per la produttività, il PIL e l'output gap
    print(f"Produttività media: {hetero_results['crescita_produttivita'].mean():.2f}%")
    print(f"Produttività std: {hetero_results['crescita_produttivita'].std():.2f}%")
    print(f"PIL potenziale medio: {hetero_results['pil_potenziale'].mean():.2f}%")
    print(f"PIL potenziale std: {hetero_results['pil_potenziale'].std():.2f}%")
    print(f"Output gap medio: {hetero_results['output_gap'].mean():.2f}%")
    print(f"Output gap std: {hetero_results['output_gap'].std():.2f}%")

    print(f"Inflazione media: {hetero_results['inflazione'].mean():.2f}%")
    print(f"Inflazione std: {hetero_results['inflazione'].std():.2f}%")

    # Squilibrio del mercato del credito
    credit_imbalance = hetero_results['domanda_fondi'] - hetero_results['offerta_fondi']
    print(f"Squilibrio del mercato del credito medio: {credit_imbalance.mean():.2f}")
    print(f"Squilibrio del mercato del credito std: {credit_imbalance.std():.2f}")    
    # Grafici
    hetero_model.plot_heterogeneous_results(hetero_results)

    # Grafico delle dinamiche economiche
    plt.figure(figsize=(12, 6))
    plt.plot(hetero_results['periodo'], hetero_results['pil_totale'], label='PIL Totale', color='blue')
    plt.plot(hetero_results['periodo'], hetero_results['pil_potenziale'], label='PIL Potenziale', color='orange')
    plt.fill_between(hetero_results['periodo'], 
                     hetero_results['pil_totale'], 
                     hetero_results['pil_potenziale'], 
                     color='lightgray', alpha=0.5, label='Output Gap')
    plt.title('Dinamiche Economiche: PIL Totale vs PIL Potenziale')
    plt.xlabel('Periodo')
    plt.ylabel('PIL (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


    # Grafico delle dinamiche inflazionistiche distinguendo le componenti
    plt.figure(figsize=(12, 6))
    
    # Grafico a barre sovrapposte per le componenti
    plt.bar(hetero_results['periodo'], hetero_results['pi_supply'], 
            label='Componente Offerta', color='lightblue', alpha=0.7)
    plt.bar(hetero_results['periodo'], hetero_results['pi_demand'], 
            bottom=hetero_results['pi_supply'],
            label='Componente Domanda', color='salmon', alpha=0.7)
    plt.bar(hetero_results['periodo'], hetero_results['pi_financial'], 
            bottom=hetero_results['pi_supply'] + hetero_results['pi_demand'],
            label='Componente Finanziaria', color='lightgreen', alpha=0.7)
    
    # Inflazione totale come linea sopra le barre
    plt.plot(hetero_results['periodo'], hetero_results['inflazione'], 
             label='Inflazione Totale', color='black', linewidth=2, marker='o', markersize=3)
    
    plt.title('Decomposizione dell\'Inflazione')
    plt.xlabel('Periodo')
    plt.ylabel('Inflazione (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Printa i dati storici del tasso naturale e del tasso di mercato
    print("\nStorico tasso naturale e tasso di mercato:")
    print(hetero_results[['periodo', 'tasso_naturale', 'tasso_mercato']].head(10))


    # Salva risultati
    hetero_results.to_csv('heterogeneous_agent_simulation.csv', index=False)
    print("\nRisultati salvati in 'heterogeneous_agent_simulation.csv'")
