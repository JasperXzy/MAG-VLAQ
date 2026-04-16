from models_baseline.dbvanilla2d import DBVanilla2D
from network_mm.mm import MM


def build_models(args):
    db_model = DBVanilla2D(mode="db", dim=args.features_dim, args=args)
    query_model = MM(args=args)
    return db_model, query_model
