import React from 'react';
import { motion } from 'framer-motion';

export default function AnimatedHero() {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2,
        delayChildren: 0.3,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.8, ease: 'easeOut' },
    },
  };

  return (
    <motion.section
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="relative py-32 px-4 text-center overflow-hidden"
    >
      {/* Decorative elements */}
      <motion.div
        animate={{ y: [0, -20, 0] }}
        transition={{ duration: 4, repeat: Infinity }}
        className="absolute top-10 right-10 text-6xl opacity-20"
      >
        ✨
      </motion.div>
      <motion.div
        animate={{ y: [0, 20, 0] }}
        transition={{ duration: 5, repeat: Infinity }}
        className="absolute bottom-10 left-10 text-6xl opacity-20"
      >
        🌸
      </motion.div>

      <motion.h1 variants={itemVariants} className="text-6xl md:text-7xl font-serif font-bold text-muse-dark mb-4">
        🌸 Muse.me
      </motion.h1>

      <motion.h2 variants={itemVariants} className="text-3xl md:text-4xl font-serif text-muse-rose mb-6">
        Your Life, Romanticized by AI
      </motion.h2>

      <motion.p variants={itemVariants} className="text-xl text-gray-700 font-serif max-w-2xl mx-auto mb-4">
        Transform your mundane journal entries, bios, or daily routines into aesthetic alter egos—complete with identity titles, fictional schedules, personality traits, moodboard prompts, Spotify playlists, and downloadable visual identity cards.
      </motion.p>

      <motion.div variants={itemVariants} className="text-lg text-muse-rose font-serif italic">
        💡 Because life is more beautiful when viewed through an aesthetic lens.
      </motion.div>
    </motion.section>
  );
}
