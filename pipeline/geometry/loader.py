import os
import json
import geopandas as gpd
from shapely.geometry import box, mapping
import xml.etree.ElementTree as ET
from typing import Tuple, Dict, Any, Union
import ee


def kml_to_geojson(kml_path: str, output_path: str = None) -> str:
    """
    Convert KML file to GeoJSON format.
    
    Args:
        kml_path: Path to KML file
        output_path: Optional output path for GeoJSON file
        
    Returns:
        Path to created GeoJSON file
    """
    try:
        # Use geopandas to read KML and convert to GeoJSON
        gdf = gpd.read_file(kml_path, driver="KML")
        
        # Generate output path if not provided
        if output_path is None:
            base_name = os.path.splitext(kml_path)[0]
            output_path = f"{base_name}.geojson"
        
        # Save as GeoJSON
        gdf.to_file(output_path, driver="GeoJSON")
        
        print(f"Successfully converted {kml_path} to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error converting KML to GeoJSON: {e}")
        # Fallback to manual parsing
        return _manual_kml_to_geojson(kml_path, output_path)


def _manual_kml_to_geojson(kml_path: str, output_path: str) -> str:
    """
    Manual KML to GeoJSON conversion as fallback.
    """
    tree = ET.parse(kml_path)
    root = tree.getroot()
    
    # Handle namespace
    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    
    features = []
    
    # Find all Placemark elements
    for placemark in root.findall(".//kml:Placemark", ns):
        properties = {}
        
        # Extract name
        name_elem = placemark.find("kml:name", ns)
        if name_elem is not None:
            properties["name"] = name_elem.text
            
        # Extract description
        desc_elem = placemark.find("kml:description", ns)
        if desc_elem is not None:
            properties["description"] = desc_elem.text
            
        # Extract coordinates from Polygon or Point
        coords_elem = placemark.find(".//kml:coordinates", ns)
        if coords_elem is not None:
            coords_text = coords_elem.text.strip()
            coords = []
            for coord in coords_text.split():
                lon, lat, *alt = coord.split(',')
                coords.append([float(lon), float(lat)])
            
            # Create polygon geometry
            geometry = {
                "type": "Polygon",
                "coordinates": [coords]
            }
            
            feature = {
                "type": "Feature",
                "properties": properties,
                "geometry": geometry
            }
            features.append(feature)
    
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    # Write to file
    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)
    
    return output_path


def load_geometry(file_path: str) -> Tuple[Any, Tuple[float, float, float, float], Dict]:
    """
    Load geometry from KML or GeoJSON file and return standardized format.
    
    Args:
        file_path: Path to KML or GeoJSON file
        
    Returns:
        Tuple of (ee.Geometry, bbox_coords, metadata)
        - ee.Geometry: Earth Engine geometry object
        - bbox_coords: (minx, miny, maxx, maxy) in WGS84
        - metadata: Dictionary with geometry metadata
    """
    # Convert KML to GeoJSON if needed
    if file_path.lower().endswith('.kml'):
        print(f"Converting KML to GeoJSON: {file_path}")
        geojson_path = kml_to_geojson(file_path)
    elif file_path.lower().endswith(('.geojson', '.json')):
        geojson_path = file_path
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    # Load with geopandas for geometry processing
    try:
        gdf = gpd.read_file(geojson_path)
        
        # Ensure WGS84 projection
        if gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        
        # Get union of all geometries
        geometry = gdf.unary_union
        
        # Get bounding box
        minx, miny, maxx, maxy = geometry.bounds
        bbox_coords = (minx, miny, maxx, maxy)
        
        # Convert to GeoJSON for Earth Engine
        if hasattr(geometry, 'geoms'):
            # MultiPolygon or GeometryCollection
            geom_list = []
            for geom in geometry.geoms:
                geom_list.append(mapping(geom))
            geojson_geom = {"type": "GeometryCollection", "geometries": geom_list}
        else:
            # Single geometry
            geojson_geom = mapping(geometry)
        
        # Create Earth Engine geometry
        ee_geometry = ee.Geometry(geojson_geom)
        
        # Metadata
        metadata = {
            "num_features": len(gdf),
            "area_km2": geometry.area * 111 * 111,  # Rough conversion to km²
            "centroid": [geometry.centroid.x, geometry.centroid.y],
            "bbox": bbox_coords,
            "crs": "EPSG:4326"
        }
        
        print(f"Loaded geometry with {metadata['num_features']} features")
        print(f"Area: {metadata['area_km2']:.2f} km²")
        print(f"Bbox: {bbox_coords}")
        
        return ee_geometry, bbox_coords, metadata
        
    except Exception as e:
        print(f"Error loading geometry: {e}")
        raise


def create_buffered_geometry(ee_geometry: Any, buffer_meters: float = 1000) -> Any:
    """
    Create a buffered version of the geometry for data download.
    
    Args:
        ee_geometry: Earth Engine geometry object
        buffer_meters: Buffer distance in meters
        
    Returns:
        Buffered Earth Engine geometry
    """
    return ee_geometry.buffer(buffer_meters)


def validate_geometry_size(bbox_coords: Tuple[float, float, float, float], 
                          max_area_km2: float = 10000) -> bool:
    """
    Validate that geometry is not too large for processing.
    
    Args:
        bbox_coords: (minx, miny, maxx, maxy)
        max_area_km2: Maximum allowed area in km²
        
    Returns:
        True if geometry is valid size
    """
    minx, miny, maxx, maxy = bbox_coords
    width_deg = maxx - minx
    height_deg = maxy - miny
    
    # Rough conversion to km² (1 degree ≈ 111 km)
    area_km2 = width_deg * height_deg * 111 * 111
    
    if area_km2 > max_area_km2:
        print(f"Warning: Geometry area ({area_km2:.2f} km²) exceeds maximum ({max_area_km2} km²)")
        return False
    
    return True