import { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [selectedUser, setSelectedUser] = useState('');
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Predefined user IDs
  const userIds = ['6248', '9785', '2346'];
  
  // This is a placeholder sequence
  const dummySequence = [333, 343, 4415, 534, 452];

  // TMDb API configuration - Add your API key here
  const TMDB_API_KEY = 'fcf1468f0c45f856fecb7089d19e98ac'; // Add your TMDb API key here
  const TMDB_SEARCH_URL = 'https://api.themoviedb.org/3/search/movie';
  const TMDB_IMAGE_BASE_URL = 'https://image.tmdb.org/t/p/w500';
  
  // API endpoint
  const API_ENDPOINT = 'http://127.0.0.1:8000/predict';

  useEffect(() => {
    if (selectedUser) {
      fetchRecommendations(selectedUser);
    }
  }, [selectedUser]);

  const fetchRecommendations = async (userId) => {
    setLoading(true);
    setError(null);
    
    // Log the request details
    const requestBody = {
      user_id: parseInt(userId),
      sequence: dummySequence,
      top_k: 10
    };
    // console.log('API Request to:', API_ENDPOINT);
    // console.log('Request Body:', JSON.stringify(requestBody, null, 2));
    
    try {
      const response = await fetch(API_ENDPOINT, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });
      
      if (!response.ok) {
        let errorBody = 'Failed to fetch recommendations';
        try {
          const responseText = await response.text();
          try {
            const errData = JSON.parse(responseText);
            if (errData && errData.error) {
              errorBody = errData.error;
            }
          } catch (jsonParseError) {
            errorBody = responseText || errorBody;
          }
        } catch (textError) {
          console.error('Failed to read response text:', textError);
        }
        
        throw new Error(`API Error (${response.status}): ${errorBody}`);
      }
      
      const data = await response.json();
      
      if (!data || !data.top_items) {
        throw new Error('Invalid API response format: missing top_items');
      }
      
      // Get movie details from TMDb
      if (TMDB_API_KEY) {
        const moviesWithDetails = await fetchMovieDetails(data.top_items);
        setRecommendations(moviesWithDetails);
      } else {
        // If no TMDb API key, just show the movie IDs
        const formattedRecommendations = data.top_items.map(movieId => ({
          id: movieId,
          title: `Movie ${movieId}`
        }));
        setRecommendations(formattedRecommendations);
      }

    } catch (error) {
      console.error('Error fetching recommendations:', error);
      setError(error.message || 'Failed to get recommendations. Check API.');
      setRecommendations([]); // Clear recommendations on error
    } finally {
      setLoading(false);
    }
  };

  // Fetch movie details from TMDb
  const fetchMovieDetails = async (movieIds) => {
    if (!TMDB_API_KEY) return movieIds.map(id => ({ id, title: `Movie ${id}` }));
    
    console.log('===== TMDb API Debug =====');
    console.log(`Fetching details for ${movieIds.length} movies:`, movieIds);
    
    // For each movie ID, search in TMDb to find details
    const moviePromises = movieIds.map(async (movieId) => {
      try {
        // Add a mapping table for common MovieLens IDs to movie titles
        // This improves search accuracy on TMDb
        const movieMap = {
          1: "Toy Story",
          2: "Jumanji",
          3: "Grumpier Old Men",
          4: "Waiting to Exhale",
          5: "Father of the Bride Part II",
          6: "Heat",
          7: "Sabrina",
          8: "Tom and Huck",
          9: "Sudden Death",
          10: "GoldenEye",
          // Add more mappings as needed
        };
        
        // Use a mapped title if available, otherwise search by "Movie {id}"
        const searchQuery = movieMap[movieId] || `Movie ${movieId}`;
        
        // Try up to 3 different search strategies if needed
        let movie = null;
        let tmdbError = null;
        
        // First try: Search by mapped title or "Movie ID"
        try {
          console.log(`[Movie ${movieId}] Searching TMDb for: "${searchQuery}"`);
          movie = await searchTMDB(searchQuery, movieId);
        } catch (error) {
          console.warn(`[Movie ${movieId}] First search attempt failed:`, error.message);
          tmdbError = error;
        }
        
        // Second try: If the movie ID is numeric and no results, try just the ID
        if (!movie && !isNaN(movieId)) {
          try {
            console.log(`[Movie ${movieId}] Second attempt - searching by just ID: "${movieId}"`);
            movie = await searchTMDB(movieId.toString(), movieId);
          } catch (retryError) {
            console.warn(`[Movie ${movieId}] Second search attempt failed:`, retryError.message);
          }
        }
        
        // Third try: Add "film" to the search term
        if (!movie) {
          try {
            const filmQuery = `${searchQuery} film`;
            console.log(`[Movie ${movieId}] Third attempt - adding "film" to query: "${filmQuery}"`);
            movie = await searchTMDB(filmQuery, movieId);
          } catch (finalError) {
            console.warn(`[Movie ${movieId}] Final search attempt failed:`, finalError.message);
          }
        }
        
        // If we found a movie, return its details
        if (movie) {
          return {
            ...movie,
            id: movieId,  // Keep the original MovieLens ID
          };
        }
        
        // If all search attempts failed
        console.error(`[Movie ${movieId}] All TMDb search attempts failed`);
        return { 
          id: movieId, 
          title: movieMap[movieId] || `Movie ${movieId}`, 
          poster_path: null,
          tmdb_error: tmdbError ? tmdbError.message : "No matches found"
        };
        
      } catch (error) {
        console.error(`[Movie ${movieId}] Unexpected error in fetch process:`, error);
        console.error(`[Movie ${movieId}] Error details:`, {
          name: error.name,
          message: error.message,
          stack: error.stack
        });
        return { 
          id: movieId, 
          title: `Movie ${movieId}`, 
          poster_path: null,
          tmdb_error: error.message
        };
      }
    });
    
    const results = await Promise.all(moviePromises);
    const successCount = results.filter(movie => movie.poster_path).length;
    console.log(`TMDb lookup complete. Found posters for ${successCount}/${movieIds.length} movies.`);
    console.log('=========================');
    return results;
  };

  // Helper function to search TMDb
  const searchTMDB = async (query, movieId) => {
    const searchUrl = `${TMDB_SEARCH_URL}?api_key=${TMDB_API_KEY}&query=${encodeURIComponent(query)}`;
    
    try {
      // Fetch data from TMDb
      const response = await fetch(searchUrl);
      console.log(`[Movie ${movieId}] TMDb response status:`, response.status, response.statusText);
      
      if (!response.ok) {
        throw new Error(`TMDb API error: ${response.status}`);
      }
      
      // Get the raw response text first
      const responseText = await response.text();
      console.log(`[Movie ${movieId}] TMDb raw response:`, responseText.substring(0, 200) + '...');
      
      // Then parse it as JSON
      const data = JSON.parse(responseText);
      console.log(`[Movie ${movieId}] TMDb results count:`, data.results ? data.results.length : 0);
      
      // If we get search results, use the first one (most relevant)
      if (data.results && data.results.length > 0) {
        const movie = data.results[0];
        console.log(`[Movie ${movieId}] TMDb match found:`, {
          movieId: movieId,
          tmdbId: movie.id,
          title: movie.title,
          originalQuery: query,
          hasPoster: !!movie.poster_path,
          releaseDate: movie.release_date
        });
        
        return {
          title: movie.title,
          poster_path: movie.poster_path,
          overview: movie.overview,
          release_date: movie.release_date,
          vote_average: movie.vote_average,
          tmdb_id: movie.id
        };
      }
      
      console.warn(`[Movie ${movieId}] No TMDb results found for query "${query}"`);
      return null;
    } catch (error) {
      console.error(`[Movie ${movieId}] TMDb search error for query "${query}":`, error);
      throw error;
    }
  };

  const handleUserChange = (e) => {
    setSelectedUser(e.target.value);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Movie Recommendations</h1>
        
        <div className="user-selector">
          <label htmlFor="user-select">Select a User: </label>
          <select 
            id="user-select" 
            value={selectedUser} 
            onChange={handleUserChange}
          >
            <option value="">-- Select User --</option>
            {userIds.map(id => (
              <option key={id} value={id}>{id}</option>
            ))}
          </select>
        </div>
        
        {error && (
          <div className="error-message">
            {error}
          </div>
        )}
        
        {selectedUser && (
          <div className="recommendations-container">
            <h2>Top Recommendation IDs for User {selectedUser}</h2>
            <p className="sequence-info">Based on sequence: [{dummySequence.join(', ')}]</p>
            
            {loading ? (
              <div className="loading">
                <div className="loading-spinner"></div>
                <p>Loading recommendations...</p>
              </div>
            ) : (
              <div className="movie-grid">
                {recommendations.map(movie => (
                  <div key={movie.id} className="movie-card">
                    <div className="movie-image">
                      {movie.poster_path && TMDB_API_KEY ? (
                        <img 
                          src={`${TMDB_IMAGE_BASE_URL}${movie.poster_path}`} 
                          alt={movie.title} 
                        />
                      ) : (
                        <div className="movie-image-placeholder">
                          ID: {movie.id}
                          {movie.tmdb_error && (
                            <div className="tmdb-error-hint">
                              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <circle cx="12" cy="12" r="10"></circle>
                                <line x1="12" y1="8" x2="12" y2="12"></line>
                                <line x1="12" y1="16" x2="12.01" y2="16"></line>
                              </svg>
                              <span className="tooltip-text">{movie.tmdb_error}</span>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                    <div className="movie-title">{movie.title}</div>
                    {movie.release_date && (
                      <div className="movie-year">
                        {new Date(movie.release_date).getFullYear()}
                        {movie.vote_average && ` • ${movie.vote_average.toFixed(1)}/10`}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </header>
    </div>
  );
}

export default App;
