/**
 * Archetype dataset for persona generation
 */
export const ARCHETYPE_DATASET = [
  {
    id: 'cyberpunk_poet',
    aesthetic: 'Cyberpunk Poet',
    traits: ['Melancholic', 'Tech-savvy', 'Nocturnal', 'Introspective'],
    vibe_keywords: ['neon', 'dystopian', 'rebellious', 'poetic'],
    moodboard_prompts: [
      'neon skyline at midnight, rain reflecting lights',
      'vintage CRT monitor glowing in darkness',
      'punk jacket covered in flowers and tech',
      'holographic interface with poetry text',
      'dark city rooftop with stars visible through smog',
      'broken keyboard with moss growing',
      'cyberpunk femme fatale with sad eyes'
    ],
    playlist_keywords: ['dark', 'electronic', 'melancholic', 'cyberpunk']
  },
  {
    id: 'cottagecore_romantic',
    aesthetic: 'Cottagecore Romantic',
    traits: ['Whimsical', 'Nature-loving', 'Gentle', 'Nostalgic'],
    vibe_keywords: ['cottage', 'pastoral', 'vintage', 'cottagecore'],
    moodboard_prompts: [
      'overgrown garden cottage with wildflowers',
      'vintage tea set in morning sunlight',
      'handwritten journal surrounded by flowers',
      'forest path lined with mushrooms',
      'misty morning in a small village',
      'antique lace and dried herbs',
      'cozy fireplace with books and candles'
    ],
    playlist_keywords: ['acoustic', 'folk', 'indie', 'cottagecore']
  },
  {
    id: 'cloudcore_dreamer',
    aesthetic: 'Cloudcore Catnapper',
    traits: ['Dreamy', 'Introspective', 'Gentle', 'Artistic'],
    vibe_keywords: ['fluffy', 'cloud', 'pastel', 'dreamy'],
    moodboard_prompts: [
      'fluffy clouds against pink sky',
      'soft pastel sunrise through window',
      'cozy blankets and pillows in warm light',
      'cat sleeping among clouds',
      'cotton candy colored landscape',
      'soft focus photography aesthetic',
      'dreamy underwater or cloud realm'
    ],
    playlist_keywords: ['lofi', 'ambient', 'calm', 'dreamy']
  },
  {
    id: 'dark_academia',
    aesthetic: 'Dark Academia Rebel',
    traits: ['Intellectual', 'Mysterious', 'Ambitious', 'Bookish'],
    vibe_keywords: ['academia', 'dark', 'mysterious', 'gothic'],
    moodboard_prompts: [
      'old library with candlelit bookshelves',
      'leather-bound books and fountain pens',
      'gothic university architecture',
      'wine glasses and classical art',
      'mysterious figure in vintage school uniform',
      'ornate vintage textbooks',
      'candlelit study with autumn leaves'
    ],
    playlist_keywords: ['classical', 'dark', 'mysterious', 'indie']
  },
  {
    id: 'maximalist_artist',
    aesthetic: 'Maximalist Creator',
    traits: ['Expressive', 'Colorful', 'Bold', 'Unapologetic'],
    vibe_keywords: ['colorful', 'bold', 'artistic', 'expressive'],
    moodboard_prompts: [
      'vibrant art studio with exploding colors',
      'mixed media collage bursting with life',
      'bold makeup and clashing patterns',
      'graffiti walls and street art energy',
      'cluttered creative space filled with inspiration',
      'rainbow gradient aesthetic',
      'experimental fashion and art pieces'
    ],
    playlist_keywords: ['experimental', 'energetic', 'bold', 'electronic']
  }
];

/**
 * Extract keywords from user input
 */
export function extractKeywords(text) {
  const stopWords = new Set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your', 'his', 'her', 'its', 'our', 'their']);
  
  const words = text.toLowerCase()
    .replace(/[^\w\s]/g, ' ')
    .split(/\s+/)
    .filter(word => word.length > 3 && !stopWords.has(word));
  
  return [...new Set(words)];
}

/**
 * Retrieve relevant archetypes based on keywords
 */
export function retrieveArchetypes(keywords, aestheticPreference = null, limit = 2) {
  let scoredArchetypes = ARCHETYPE_DATASET.map(archetype => {
    let score = 0;
    
    // Check keyword matches
    keywords.forEach(keyword => {
      if (archetype.vibe_keywords.some(vibe => vibe.includes(keyword))) {
        score += 2;
      }
      if (archetype.traits.some(trait => trait.toLowerCase().includes(keyword))) {
        score += 1.5;
      }
      if (archetype.aesthetic.toLowerCase().includes(keyword)) {
        score += 3;
      }
    });
    
    // Boost score if matches aesthetic preference
    if (aestheticPreference && archetype.aesthetic.includes(aestheticPreference)) {
      score += 5;
    }
    
    return { archetype, score };
  });
  
  // Sort by score and return top archetypes
  scoredArchetypes.sort((a, b) => b.score - a.score);
  return scoredArchetypes.slice(0, limit).map(item => item.archetype);
}
