from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import uuid
import uvicorn
from datetime import datetime
import logging
import httpx

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Création de l'application FastAPI
app = FastAPI(
    title="Portfolio Management Service",
    description="Service de gestion de portfolios pour l'analyse Monte Carlo",
    version="1.0.0"
)

# Configuration des services externes
SIMULATION_SERVICE_URL = "http://simulation-service:8001"

# Modèles de données
class Asset(BaseModel):
    symbol: str
    name: str
    weight: float
    expected_return: float
    volatility: float
    sector: Optional[str] = None

class Portfolio(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    assets: List[Asset]
    correlation_matrix: List[List[float]]
    created_at: str
    updated_at: str
    risk_profile: str = "moderate"  # conservative, moderate, aggressive

class PortfolioCreate(BaseModel):
    name: str
    description: Optional[str] = None
    assets: List[Asset]
    correlation_matrix: List[List[float]]
    risk_profile: str = "moderate"

class PortfolioUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    assets: Optional[List[Asset]] = None
    correlation_matrix: Optional[List[List[float]]] = None
    risk_profile: Optional[str] = None

class SimulationRequest(BaseModel):
    portfolio_id: str
    initial_value: float = 100000
    time_horizon: int = 252
    num_simulations: int = 1000

# Stockage en mémoire (en production, utiliser une vraie base de données)
portfolios_db: Dict[str, Portfolio] = {}

# Portfolios pré-configurés pour les tests
def init_sample_portfolios():
    """Initialise quelques portfolios d'exemple"""
    
    # Portfolio conservateur
    conservative_assets = [
        Asset(
            symbol="BND",
            name="Obligations US",
            weight=0.6,
            expected_return=0.04,
            volatility=0.08,
            sector="Fixed Income"
        ),
        Asset(
            symbol="VTI",
            name="Actions US Total Market",
            weight=0.3,
            expected_return=0.08,
            volatility=0.16,
            sector="Equity"
        ),
        Asset(
            symbol="VTIAX",
            name="Actions Internationales",
            weight=0.1,
            expected_return=0.07,
            volatility=0.18,
            sector="International Equity"
        )
    ]
    
    conservative_corr = [
        [1.0, 0.1, 0.05],
        [0.1, 1.0, 0.7],
        [0.05, 0.7, 1.0]
    ]
    
    # Portfolio agressif
    aggressive_assets = [
        Asset(
            symbol="QQQ",
            name="Nasdaq 100",
            weight=0.4,
            expected_return=0.15,
            volatility=0.25,
            sector="Technology"
        ),
        Asset(
            symbol="TSLA",
            name="Tesla",
            weight=0.3,
            expected_return=0.20,
            volatility=0.40,
            sector="Technology"
        ),
        Asset(
            symbol="BTC-USD",
            name="Bitcoin",
            weight=0.2,
            expected_return=0.25,
            volatility=0.60,
            sector="Cryptocurrency"
        ),
        Asset(
            symbol="NVDA",
            name="Nvidia",
            weight=0.1,
            expected_return=0.18,
            volatility=0.35,
            sector="Technology"
        )
    ]
    
    aggressive_corr = [
        [1.0, 0.6, 0.3, 0.7],
        [0.6, 1.0, 0.2, 0.8],
        [0.3, 0.2, 1.0, 0.3],
        [0.7, 0.8, 0.3, 1.0]
    ]
    
    # Créer les portfolios
    portfolios = [
        Portfolio(
            id="conservative-001",
            name="Portfolio Conservateur",
            description="Portfolio équilibré avec focus sur la préservation du capital",
            assets=conservative_assets,
            correlation_matrix=conservative_corr,
            risk_profile="conservative",
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        ),
        Portfolio(
            id="aggressive-001", 
            name="Portfolio Agressif Tech",
            description="Portfolio haute croissance axé sur la technologie",
            assets=aggressive_assets,
            correlation_matrix=aggressive_corr,
            risk_profile="aggressive",
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
    ]
    
    for portfolio in portfolios:
        portfolios_db[portfolio.id] = portfolio
    
    logger.info(f"Initialisé {len(portfolios)} portfolios d'exemple")

# Fonctions utilitaires
def validate_portfolio_weights(assets: List[Asset]) -> bool:
    """Valide que les poids totalisent 1.0"""
    total_weight = sum(asset.weight for asset in assets)
    return abs(total_weight - 1.0) < 0.01

def validate_correlation_matrix(matrix: List[List[float]], n_assets: int) -> bool:
    """Valide la matrice de corrélation"""
    if len(matrix) != n_assets:
        return False
    
    for i, row in enumerate(matrix):
        if len(row) != n_assets:
            return False
        if abs(row[i] - 1.0) > 0.01:  # Diagonal doit être 1
            return False
        for j, corr in enumerate(row):
            if abs(corr - matrix[j][i]) > 0.01:  # Symétrie
                return False
            if abs(corr) > 1.0:  # Valeurs valides
                return False
    
    return True

async def call_simulation_service(portfolio: Portfolio, request: SimulationRequest):
    """Appelle le service de simulation Monte Carlo"""
    try:
        simulation_payload = {
            "assets": [asset.dict() for asset in portfolio.assets],
            "correlation_matrix": portfolio.correlation_matrix,
            "initial_value": request.initial_value,
            "time_horizon": request.time_horizon,
            "num_simulations": request.num_simulations
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{SIMULATION_SERVICE_URL}/simulate",
                json=simulation_payload
            )
            response.raise_for_status()
            return response.json()
            
    except httpx.RequestError as e:
        logger.error(f"Erreur de connexion au service de simulation: {e}")
        raise HTTPException(
            status_code=503, 
            detail="Service de simulation indisponible"
        )
    except httpx.HTTPStatusError as e:
        logger.error(f"Erreur du service de simulation: {e}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Erreur simulation: {e.response.text}"
        )

# Routes API
@app.on_event("startup")
async def startup_event():
    """Initialisation au démarrage"""
    init_sample_portfolios()
    logger.info("Portfolio service démarré")

@app.get("/")
async def root():
    return {
        "service": "Portfolio Management Service",
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "portfolio-service",
        "portfolios_count": len(portfolios_db)
    }

@app.get("/portfolios", response_model=List[Portfolio])
async def list_portfolios():
    """Liste tous les portfolios"""
    return list(portfolios_db.values())

@app.get("/portfolios/{portfolio_id}", response_model=Portfolio)
async def get_portfolio(portfolio_id: str):
    """Récupère un portfolio par son ID"""
    if portfolio_id not in portfolios_db:
        raise HTTPException(status_code=404, detail="Portfolio non trouvé")
    
    return portfolios_db[portfolio_id]

@app.post("/portfolios", response_model=Portfolio)
async def create_portfolio(portfolio_data: PortfolioCreate):
    """Crée un nouveau portfolio"""
    
    # Validations
    if not validate_portfolio_weights(portfolio_data.assets):
        raise HTTPException(
            status_code=400, 
            detail="Les poids des actifs doivent totaliser 1.0"
        )
    
    if not validate_correlation_matrix(
        portfolio_data.correlation_matrix, 
        len(portfolio_data.assets)
    ):
        raise HTTPException(
            status_code=400,
            detail="Matrice de corrélation invalide"
        )
    
    # Créer le portfolio
    portfolio_id = str(uuid.uuid4())
    portfolio = Portfolio(
        id=portfolio_id,
        name=portfolio_data.name,
        description=portfolio_data.description,
        assets=portfolio_data.assets,
        correlation_matrix=portfolio_data.correlation_matrix,
        risk_profile=portfolio_data.risk_profile,
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat()
    )
    
    portfolios_db[portfolio_id] = portfolio
    logger.info(f"Portfolio créé: {portfolio_id}")
    
    return portfolio

@app.put("/portfolios/{portfolio_id}", response_model=Portfolio)
async def update_portfolio(portfolio_id: str, update_data: PortfolioUpdate):
    """Met à jour un portfolio"""
    
    if portfolio_id not in portfolios_db:
        raise HTTPException(status_code=404, detail="Portfolio non trouvé")
    
    portfolio = portfolios_db[portfolio_id]
    
    # Mise à jour des champs
    if update_data.name is not None:
        portfolio.name = update_data.name
    if update_data.description is not None:
        portfolio.description = update_data.description
    if update_data.risk_profile is not None:
        portfolio.risk_profile = update_data.risk_profile
    
    if update_data.assets is not None:
        if not validate_portfolio_weights(update_data.assets):
            raise HTTPException(
                status_code=400,
                detail="Les poids des actifs doivent totaliser 1.0"
            )
        portfolio.assets = update_data.assets
    
    if update_data.correlation_matrix is not None:
        n_assets = len(portfolio.assets)
        if not validate_correlation_matrix(update_data.correlation_matrix, n_assets):
            raise HTTPException(
                status_code=400,
                detail="Matrice de corrélation invalide"
            )
        portfolio.correlation_matrix = update_data.correlation_matrix
    
    portfolio.updated_at = datetime.now().isoformat()
    portfolios_db[portfolio_id] = portfolio
    
    logger.info(f"Portfolio mis à jour: {portfolio_id}")
    return portfolio

@app.delete("/portfolios/{portfolio_id}")
async def delete_portfolio(portfolio_id: str):
    """Supprime un portfolio"""
    
    if portfolio_id not in portfolios_db:
        raise HTTPException(status_code=404, detail="Portfolio non trouvé")
    
    del portfolios_db[portfolio_id]
    logger.info(f"Portfolio supprimé: {portfolio_id}")
    
    return {"message": f"Portfolio {portfolio_id} supprimé"}

@app.post("/portfolios/{portfolio_id}/simulate")
async def simulate_portfolio(portfolio_id: str, request: SimulationRequest):
    """Lance une simulation Monte Carlo pour un portfolio"""
    
    if portfolio_id not in portfolios_db:
        raise HTTPException(status_code=404, detail="Portfolio non trouvé")
    
    portfolio = portfolios_db[portfolio_id]
    request.portfolio_id = portfolio_id
    
    logger.info(f"Lancement simulation pour portfolio {portfolio_id}")
    
    # Appeler le service de simulation
    result = await call_simulation_service(portfolio, request)
    
    # Ajouter les métadonnées du portfolio au résultat
    result["portfolio_info"] = {
        "id": portfolio.id,
        "name": portfolio.name,
        "risk_profile": portfolio.risk_profile,
        "assets_count": len(portfolio.assets)
    }
    
    return result

@app.get("/portfolios/{portfolio_id}/summary")
async def get_portfolio_summary(portfolio_id: str):
    """Résumé détaillé d'un portfolio"""
    
    if portfolio_id not in portfolios_db:
        raise HTTPException(status_code=404, detail="Portfolio non trouvé")
    
    portfolio = portfolios_db[portfolio_id]
    
    # Calcul de statistiques de base
    total_weight = sum(asset.weight for asset in portfolio.assets)
    avg_expected_return = sum(asset.expected_return * asset.weight for asset in portfolio.assets)
    weighted_volatility = sum(asset.volatility * asset.weight for asset in portfolio.assets)
    
    sectors = {}
    for asset in portfolio.assets:
        sector = asset.sector or "Other"
        if sector not in sectors:
            sectors[sector] = 0
        sectors[sector] += asset.weight
    
    return {
        "portfolio": portfolio,
        "statistics": {
            "total_weight": round(total_weight, 4),
            "weighted_expected_return": round(avg_expected_return, 4),
            "weighted_volatility": round(weighted_volatility, 4),
            "assets_count": len(portfolio.assets),
            "sector_allocation": sectors
        }
    }

# Point d'entrée
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info"
    )
