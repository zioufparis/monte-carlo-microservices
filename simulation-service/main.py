from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import numpy as np
import uvicorn
from datetime import datetime
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Création de l'application FastAPI
app = FastAPI(
    title="Monte Carlo Simulation Service",
    description="Service de simulation Monte Carlo pour l'analyse de portefeuille",
    version="1.0.0"
)

# Modèles de données (schémas)
class Asset(BaseModel):
    symbol: str
    weight: float
    expected_return: float
    volatility: float

class SimulationRequest(BaseModel):
    assets: List[Asset]
    correlation_matrix: List[List[float]]
    initial_value: float = 100000
    time_horizon: int = 252  # jours de trading par an
    num_simulations: int = 1000

class SimulationResult(BaseModel):
    simulation_id: str
    final_values: List[float]
    percentiles: Dict[str, float]
    var_95: float
    var_99: float
    expected_value: float
    timestamp: str

# Fonction principale de simulation Monte Carlo
def run_monte_carlo_simulation(request: SimulationRequest) -> SimulationResult:
    """
    Exécute une simulation Monte Carlo pour un portefeuille d'actifs
    """
    try:
        # Préparation des données
        weights = np.array([asset.weight for asset in request.assets])
        returns = np.array([asset.expected_return for asset in request.assets])
        volatilities = np.array([asset.volatility for asset in request.assets])
        correlation_matrix = np.array(request.correlation_matrix)
        
        # Calcul de la matrice de covariance
        # Covariance = Correlation × (volatility_i × volatility_j)
        cov_matrix = correlation_matrix * np.outer(volatilities, volatilities)
        
        # Calcul des paramètres du portefeuille
        portfolio_return = np.dot(weights, returns)
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_std = np.sqrt(portfolio_variance)
        
        logger.info(f"Portefeuille: rendement={portfolio_return:.4f}, volatilité={portfolio_std:.4f}")
        
        # Génération des simulations Monte Carlo
        # Chaque simulation représente l'évolution du portefeuille sur time_horizon jours
        random_shocks = np.random.normal(0, 1, (request.num_simulations, request.time_horizon))
        
        # Calcul des rendements quotidiens
        daily_return = portfolio_return / 252  # rendement quotidien moyen
        daily_std = portfolio_std / np.sqrt(252)  # volatilité quotidienne
        
        # Simulation des prix (mouvement brownien géométrique)
        daily_returns = daily_return + daily_std * random_shocks
        
        # Calcul des valeurs finales du portefeuille
        cumulative_returns = np.cumprod(1 + daily_returns, axis=1)
        final_values = request.initial_value * cumulative_returns[:, -1]
        
        # Calcul des statistiques
        percentiles = {
            "5": float(np.percentile(final_values, 5)),
            "25": float(np.percentile(final_values, 25)),
            "50": float(np.percentile(final_values, 50)),
            "75": float(np.percentile(final_values, 75)),
            "95": float(np.percentile(final_values, 95))
        }
        
        # VaR (Value at Risk) - perte potentielle
        var_95 = request.initial_value - percentiles["5"]
        var_99 = request.initial_value - float(np.percentile(final_values, 1))
        
        # Génération d'un ID unique pour cette simulation
        simulation_id = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Simulation terminée: {len(final_values)} scénarios générés")
        
        return SimulationResult(
            simulation_id=simulation_id,
            final_values=final_values.tolist(),
            percentiles=percentiles,
            var_95=var_95,
            var_99=var_99,
            expected_value=float(np.mean(final_values)),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Erreur dans la simulation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur de simulation: {str(e)}")

# Routes API
@app.get("/")
async def root():
    """Point de santé du service"""
    return {
        "service": "Monte Carlo Simulation Service",
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Vérification de santé détaillée"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "simulation-service"
    }

@app.post("/simulate", response_model=SimulationResult)
async def create_simulation(request: SimulationRequest):
    """
    Endpoint principal: lance une simulation Monte Carlo
    """
    logger.info(f"Nouvelle demande de simulation: {len(request.assets)} actifs, {request.num_simulations} simulations")
    
    # Validation des données
    if len(request.assets) == 0:
        raise HTTPException(status_code=400, detail="Au moins un actif est requis")
    
    total_weight = sum(asset.weight for asset in request.assets)
    if abs(total_weight - 1.0) > 0.01:
        raise HTTPException(status_code=400, detail=f"Les poids doivent totaliser 1.0 (actuellement: {total_weight})")
    
    # Validation de la matrice de corrélation
    n_assets = len(request.assets)
    if len(request.correlation_matrix) != n_assets:
        raise HTTPException(status_code=400, detail="Dimension de la matrice de corrélation incorrecte")
    
    # Exécution de la simulation
    result = run_monte_carlo_simulation(request)
    return result

@app.get("/simulate/{simulation_id}")
async def get_simulation(simulation_id: str):
    """
    Récupère les résultats d'une simulation (pour l'instant, retourne un message)
    Dans un vrai système, on récupérerait depuis une base de données
    """
    return {
        "message": f"Récupération de la simulation {simulation_id}",
        "note": "Fonctionnalité à implémenter avec une base de données"
    }

# Point d'entrée pour lancer le serveur
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
