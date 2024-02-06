import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


class ResultFields:
    detection_id_field: str = "ID"
    cell_id_field: str = "EntityID"
    z_index_field: str = "ZIndex"
    geometry_field: str = "Geometry"
    entity_name_field: str = "Type"
    parent_id_field: str = "ParentID"
    parent_entity_field: str = "ParentType"
