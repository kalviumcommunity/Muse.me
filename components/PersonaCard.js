import React, { useRef } from 'react';
import { motion } from 'framer-motion';
import html2canvas from 'html2canvas';

export default function PersonaCard({ persona }) {
  const cardRef = useRef(null);

  const downloadCard = async (format = 'png') => {
    if (!cardRef.current) return;

    try {
      const canvas = await html2canvas(cardRef.current, {
        backgroundColor: '#f5e6d3',
        scale: 2,
      });

      const link = document.createElement('a');
      link.href = canvas.toDataURL(`image/${format}`);
      link.download = `${persona.persona.aesthetic_identity.replace(/\s+/g, '_')}.${format}`;
      link.click();
    } catch (error) {
      console.error('Error downloading card:', error);
    }
  };

  const containerVariants = {
    hidden: { opacity: 0, scale: 0.95 },
    visible: {
      opacity: 1,
      scale: 1,
      transition: { duration: 0.6 },
    },
  };

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="space-y-8"
    >
      {/* Card Display */}
      <motion.div
        ref={cardRef}
        className="bg-gradient-to-br from-muse-cream via-white to-muse-cream rounded-3xl shadow-2xl p-12 max-w-3xl mx-auto border-2 border-muse-rose border-opacity-20"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        {/* Header */}
        <div className="text-center mb-10 pb-8 border-b-2 border-muse-rose border-opacity-30">
          <p className="text-sm font-serif text-gray-500 tracking-widest mb-4 uppercase">
            ✨ Your Aesthetic Alter Ego ✨
          </p>
        </div>

        {/* Identity Name */}
        <h2 className="text-6xl font-serif font-bold text-muse-dark text-center mb-8 leading-tight">
          {persona.persona.aesthetic_identity}
        </h2>

        {/* Subtitle */}
        {persona.persona.aesthetic_subtitle && (
          <p className="text-2xl text-muse-rose text-center font-serif italic mb-10 leading-relaxed">
            "{persona.persona.aesthetic_subtitle}"
          </p>
        )}

        {/* Vibe Description */}
        <div className="bg-white bg-opacity-60 rounded-2xl p-8 mb-10">
          <p className="text-lg text-gray-700 text-center font-serif leading-relaxed">
            {persona.persona.vibe_description}
          </p>
        </div>

        {/* Traits */}
        <div className="mb-10">
          <h3 className="text-xl font-serif font-bold text-gray-600 tracking-widest mb-6 text-center uppercase">
            ✦ Personality Traits ✦
          </h3>
          <div className="flex flex-wrap gap-3 justify-center">
            {persona.persona.persona_traits?.map((trait, idx) => (
              <motion.span
                key={idx}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.1 + idx * 0.05 }}
                className="bg-gradient-to-r from-muse-rose to-pink-400 bg-opacity-90 text-white px-6 py-3 rounded-full font-serif text-base font-semibold shadow-md hover:shadow-lg transform hover:scale-105 transition-all"
              >
                {trait}
              </motion.span>
            ))}
          </div>
        </div>

        {/* Fictional Schedule */}
        <div className="mb-10">
          <h3 className="text-xl font-serif font-bold text-gray-600 tracking-widest mb-6 text-center uppercase">
            ✦ A Day in the Life ✦
          </h3>
          <div className="space-y-4 bg-white bg-opacity-60 rounded-2xl p-8">
            {persona.persona.fictional_schedule && Object.entries(persona.persona.fictional_schedule).map(([time, activity], idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.15 + idx * 0.05 }}
                className="flex items-start gap-4 pb-4 border-b border-gray-200 last:border-0 last:pb-0"
              >
                <span className="text-muse-rose font-serif font-bold capitalize min-w-fit text-lg">
                  {time}:
                </span>
                <p className="font-serif text-gray-700 leading-relaxed">{activity}</p>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Spotify Playlist */}
        <div className="bg-gradient-to-r from-green-50 to-emerald-50 rounded-2xl p-8 text-center border-2 border-green-200">
          <p className="text-sm font-serif text-gray-500 tracking-widest mb-3 uppercase">
            🎵 Your Soundtrack
          </p>
          <p className="text-3xl font-serif font-bold text-gray-800">
            {persona.persona.spotify_playlist}
          </p>
        </div>

        {/* Footer */}
        <div className="text-center mt-10 pt-8 border-t-2 border-muse-rose border-opacity-30">
          <p className="text-sm font-serif text-gray-500">
            Generated by Muse.me ✨ • Your life, romanticized by AI
          </p>
        </div>
      </motion.div>

      {/* Moodboard Prompt */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="card p-10 max-w-3xl mx-auto"
      >
        <h3 className="text-3xl font-serif font-bold text-muse-dark mb-6 text-center flex items-center justify-center gap-3">
          <span>🎨</span>
          <span>Moodboard Prompt</span>
        </h3>
        <div className="section-divider mb-6"></div>
        <div className="bg-gradient-to-r from-muse-cream/50 to-transparent p-8 rounded-2xl border-l-4 border-muse-rose">
          <p className="font-serif text-gray-800 text-xl leading-relaxed">
            {persona.persona.moodboard_prompt}
          </p>
        </div>
      </motion.div>

      {/* Spotify Info */}
      {persona.spotify_link && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="card p-10 max-w-3xl mx-auto text-center bg-gradient-to-br from-green-50 to-emerald-50"
        >
          <h3 className="text-3xl font-serif font-bold text-gray-800 mb-6 flex items-center justify-center gap-3">
            <span>🎵</span>
            <span>Discover Your Soundtrack</span>
          </h3>
          <div className="section-divider mb-8"></div>
          <a
            href={persona.spotify_link}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-block"
          >
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.98 }}
              className="px-12 py-5 bg-gradient-to-r from-green-500 to-emerald-600 text-white rounded-full font-serif text-lg font-bold shadow-xl hover:shadow-2xl transition-all duration-300 flex items-center gap-3 mx-auto"
            >
              <span>🎵</span>
              <span>Open on Spotify</span>
              <span>→</span>
            </motion.button>
          </a>
        </motion.div>
      )}

      {/* Download Options */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
        className="flex flex-col sm:flex-row gap-6 justify-center max-w-3xl mx-auto"
      >
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.98 }}
          onClick={() => downloadCard('png')}
          className="flex-1 px-10 py-5 bg-gradient-to-r from-muse-rose to-pink-400 text-white rounded-2xl font-serif text-lg font-bold shadow-xl hover:shadow-2xl transition-all duration-300 flex items-center justify-center gap-3"
        >
          <span>📥</span>
          <span>Download as PNG</span>
        </motion.button>
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.98 }}
          onClick={() => downloadCard('png')}
          className="flex-1 px-10 py-5 bg-white text-muse-dark border-2 border-muse-rose rounded-2xl font-serif text-lg font-bold shadow-lg hover:shadow-xl transition-all duration-300 flex items-center justify-center gap-3"
        >
          <span>🎨</span>
          <span>Share This Card</span>
        </motion.button>
      </motion.div>
    </motion.div>
  );
}
