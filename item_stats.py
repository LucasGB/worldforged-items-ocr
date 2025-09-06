from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime
from boto3.dynamodb.types import TypeSerializer

@dataclass
class DamageInfo:
    minimum_damage: Optional[int] = None
    maximum_damage: Optional[int] = None
    damage_per_second: Optional[float] = None
    speed: Optional[float] = None

@dataclass
class ItemStats:
    """Structured representation of item statistics for various item types"""
    
    # Core identification
    id: Optional[int] = None
    name: str = ""
    quality: str = ""  # Rare, Uncommon, Legendary, etc.
    item_level: Optional[int] = None
    required_level: Optional[int] = None
    
    # Item categorization
    slot: Optional[str] = None  # Head, Two-Hand, Off-Hand, Ranged, Relic, etc.
    slot_type: Optional[str] = None  # Shield, Crossbow, Polearm, Cloth, Libram, etc.
    
    # Binding and restrictions
    binding: str = ""  # Binds on Pickup, Binds to Realm, etc.
    class_restrictions: Optional[List[str]] = None
    restriction: Optional[str] = None  # For crafting requirements like "Requires Leatherworking (300)"
    
    # Base combat stats
    base_stats: Dict[str, int] = field(default_factory=dict)  # strength, agility, stamina, intellect, armor, block
    # Equipment effects and bonuses
    equip_stats: Dict[str, str] = field(default_factory=dict)  # pvePower, pvpPower, defense, hitRating, etc.
    
    # Damage information
    damage: Optional[DamageInfo] = None
    
    # Location and source
    coords: Optional[List[float]] = None  # [x, y] coordinates
    source_type: Optional[str] = None  # WorldSpawn, etc.
    zone: Optional[str] = None  # Can be extracted from partition key
    
    # Recipe/crafting specific
    teaches: Optional[str] = None  # What the recipe teaches
    crafted_item: Optional[Dict[str, Any]] = None  # Details about the crafted item
    
    # Mystic scroll specific
    description: Optional[str] = None  # Spell/ability description
    cooldown: Optional[str] = None
    
    # Metadata
    flavor_text: Optional[str] = None
    
    def __post_init__(self):
        """Initialize default values and handle backward compatibility"""
        # Ensure mutable defaults are initialized
        if self.base_stats is None:
            self.base_stats = {}
        if self.equip_stats is None:
            self.equip_stats = {}
        if self.coords is None:
            self.coords = []
    
    @classmethod
    def from_dynamodb_item(cls, item: Dict[str, Any]) -> 'ItemStats':
        """Create ItemStats instance from DynamoDB item structure"""
        
        # Extract zone from partition key
        zone = None
        if 'pk' in item and 'S' in item['pk']:
            pk_parts = item['pk']['S'].split('#')
            if len(pk_parts) > 1:
                zone = pk_parts[1]
        
        # Helper function to safely extract DynamoDB values
        def get_value(obj, key, data_type='S', default=None):
            if key in obj and data_type in obj[key]:
                if data_type == 'N':
                    return int(obj[key][data_type]) if obj[key][data_type].isdigit() else float(obj[key][data_type])
                elif data_type == 'L':
                    return [float(item['N']) for item in obj[key][data_type] if 'N' in item]
                else:
                    return obj[key][data_type]
            return default
        
        # Extract base stats
        base_stats = {}
        if 'baseStats' in item and 'M' in item['baseStats']:
            for stat, value in item['baseStats']['M'].items():
                if 'N' in value:
                    base_stats[stat] = int(value['N'])
        
        # Extract equip stats  
        equip_stats = {}
        if 'equipStats' in item and 'M' in item['equipStats']:
            for stat, value in item['equipStats']['M'].items():
                if 'S' in value:
                    equip_stats[stat] = value['S']
        
        # Extract damage info
        damage_info = item.get('damage', {})
        damage_min = None
        damage_max = None  
        damage_per_second = None
        speed = None
        
        if 'M' in damage_info:
            damage_min = get_value(damage_info['M'], 'min', 'N')
            damage_max = get_value(damage_info['M'], 'max', 'N')
            damage_per_second = get_value(damage_info['M'], 'damagePerSecond', 'N')
            speed = get_value(damage_info['M'], 'speed', 'N')
        
        # Extract coordinates
        coordinates = get_value(item, 'coords', 'L', [])
        
        # Extract class restrictions
        class_restrictions = []
        if 'classRestrictions' in item and 'L' in item['classRestrictions']:
            class_restrictions = [restriction['S'] for restriction in item['classRestrictions']['L'] if 'S' in restriction]
        
        # Extract crafted item info
        crafted_item = {}
        if 'craftedItem' in item and 'M' in item['craftedItem']:
            crafted_item_data = item['craftedItem']['M']
            crafted_item = {
                'name': get_value(crafted_item_data, 'name'),
                'quality': get_value(crafted_item_data, 'quality'),
                'required_level': get_value(crafted_item_data, 'requiredLevel', 'N'),
                'use': get_value(crafted_item_data, 'use'),
                'cooldown': get_value(crafted_item_data, 'cooldown')
            }
        
        return cls(
            id=get_value(item, 'id', 'N'),
            name=get_value(item, 'name'),
            quality=get_value(item, 'quality'),
            item_level=get_value(item, 'itemLevel', 'N'),
            required_level=get_value(item, 'requiredLevel', 'N'),
            slot=get_value(item, 'slot'),
            slot_type=get_value(item, 'slotType'),
            binding=get_value(item, 'binding'),
            class_restrictions=class_restrictions,
            restriction=get_value(item, 'restriction'),
            base_stats=base_stats,
            equip_stats=equip_stats,
            damage_min=damage_min,
            damage_max=damage_max,
            damage_per_second=damage_per_second,
            speed=speed,
            coordinates=coordinates,
            source_type=get_value(item.get('source', {}).get('M', {}), 'type') if 'source' in item else None,
            zone=zone,
            teaches=get_value(item, 'teaches'),
            crafted_item=crafted_item if crafted_item else None,
            description=get_value(item, 'description'),
            cooldown=get_value(item, 'cooldown'),
            flavor_text=get_value(item, 'flavorText'),
            created_at=get_value(item, 'created_at'),
            updated_at=get_value(item, 'updated_at')
        )
    

    def to_dynamodb_item(item: 'ItemStats'):
        serializer = TypeSerializer()
        dynamo_json = {k: serializer.serialize(v) for k, v in item.items() if v is not None}
        return dynamo_json

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'id': self.id,
            'name': self.name,
            'quality': self.quality,
            'item_level': self.item_level,
            'required_level': self.required_level,
            'slot': self.slot,
            'slot_type': self.slot_type,
            'binding': self.binding,
            'class_restrictions': self.class_restrictions,
            'restriction': self.restriction,
            'base_stats': self.base_stats,
            'equip_stats': self.equip_stats,
            'damage_range': self.damage_range,
            'damage_per_second': self.damage_per_second,
            'speed': self.speed,
            'coordinates': self.coordinates,
            'source_type': self.source_type,
            'zone': self.zone,
            'teaches': self.teaches,
            'crafted_item': self.crafted_item,
            'description': self.description,
            'cooldown': self.cooldown,
            'flavor_text': self.flavor_text,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }