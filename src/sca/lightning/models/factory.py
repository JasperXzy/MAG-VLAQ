from sca.models.dbvanilla2d import DBVanilla2D
from sca.models.mm import MM


def build_models(cfg):
    db_model = DBVanilla2D(mode="db", dim=cfg.features_dim, args=cfg)
    query_model = MM(args=cfg)
    return db_model, query_model
