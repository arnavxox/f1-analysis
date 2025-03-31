import os
import pickle
import hashlib
import logging
import time
from typing import Any, Dict, Optional, List

class F1DataCache:
    """Advanced hierarchical caching system with memory and disk layers"""
    
    def __init__(self, cache_dir: str = './f1_cache', max_memory_items: int = 1000):
        """
        Initialize the caching system
        
        Parameters:
        -----------
        cache_dir : str
            Directory to store cached files
        max_memory_items : int
            Maximum number of items to keep in memory cache
        """
        self.cache_dir = cache_dir
        self.max_memory_items = max_memory_items
        os.makedirs(cache_dir, exist_ok=True)
        self.memory_cache: Dict[str, Any] = {}
        self.cache_stats = {'hits': 0, 'misses': 0}
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("F1DataCache")

    def _get_cache_key(self, data_type: str, **kwargs) -> str:
        """Generate unique cache key using SHA-256 hashing"""
        param_str = ''.join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return hashlib.sha256(f"{data_type}_{param_str}".encode()).hexdigest()

    def get(self, data_type: str, **kwargs) -> Any:
        """Retrieve data from cache"""
        key = self._get_cache_key(data_type, **kwargs)
        
        # Memory cache lookup
        if key in self.memory_cache:
            self.cache_stats['hits'] += 1
            return self.memory_cache[key]
        
        # Disk cache lookup
        cache_path = os.path.join(self.cache_dir, f"{key}.pkl")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Manage memory cache size
                if len(self.memory_cache) >= self.max_memory_items:
                    # Remove oldest item (this is a simple approach)
                    self.memory_cache.pop(next(iter(self.memory_cache)))
                
                self.memory_cache[key] = data
                self.cache_stats['hits'] += 1
                return data
            except Exception as e:
                self.logger.error(f"Cache load error: {str(e)}")
                # Remove corrupt cache file
                try:
                    os.remove(cache_path)
                    self.logger.info(f"Removed corrupt cache file: {cache_path}")
                except:
                    pass
        
        self.cache_stats['misses'] += 1
        return None

    def set(self, data: Any, data_type: str, **kwargs) -> None:
        """Store data in cache"""
        key = self._get_cache_key(data_type, **kwargs)
        
        # Manage memory cache size
        if len(self.memory_cache) >= self.max_memory_items:
            # Remove oldest item
            self.memory_cache.pop(next(iter(self.memory_cache)))
            
        self.memory_cache[key] = data
        
        cache_path = os.path.join(self.cache_dir, f"{key}.pkl")
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            self.logger.error(f"Cache save error: {str(e)}")

    def clear(self, data_type: Optional[str] = None) -> int:
        """
        Clear cache entries
        
        Parameters:
        -----------
        data_type : str, optional
            Type of data to clear (None for all)
            
        Returns:
        --------
        int
            Number of items cleared
        """
        count = 0
        
        # Clear memory cache
        if data_type is None:
            count = len(self.memory_cache)
            self.memory_cache.clear()
        else:
            keys_to_remove = []
            for key in self.memory_cache:
                if key.startswith(data_type):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.memory_cache[key]
            count = len(keys_to_remove)
        
        # Clear disk cache
        if data_type is None:
            # Clear all files
            for file in os.listdir(self.cache_dir):
                if file.endswith('.pkl'):
                    try:
                        os.remove(os.path.join(self.cache_dir, file))
                        count += 1
                    except Exception as e:
                        self.logger.error(f"Error removing cache file {file}: {str(e)}")
        else:
            # This is a simple approach - for a more targeted approach, 
            # we would need to check each file's data_type
            pass
        
        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total = self.cache_stats['hits'] + self.cache_stats['misses']
        
        # Calculate cache size on disk
        disk_size = 0
        for file in os.listdir(self.cache_dir):
            if file.endswith('.pkl'):
                try:
                    file_path = os.path.join(self.cache_dir, file)
                    disk_size += os.path.getsize(file_path)
                except:
                    pass
        
        return {
            'hits': self.cache_stats['hits'],
            'misses': self.cache_stats['misses'],
            'hit_rate': self.cache_stats['hits'] / total if total > 0 else 0,
            'memory_items': len(self.memory_cache),
            'disk_items': len([f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]),
            'disk_size_mb': round(disk_size / (1024 * 1024), 2)
        }

    def get_cached_types(self) -> List[str]:
        """Get a list of cached data types"""
        types = set()
        for file in os.listdir(self.cache_dir):
            if file.endswith('.pkl'):
                # Extract type from filename (simplified approach)
                parts = file.split('_')
                if len(parts) > 0:
                    types.add(parts[0])
        return list(types)
