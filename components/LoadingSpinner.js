import React from 'react';
import { motion } from 'framer-motion';

export default function LoadingSpinner() {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
    >
      <motion.div
        initial={{ scale: 0.9 }}
        animate={{ scale: 1 }}
        className="bg-white rounded-xl shadow-2xl p-8 text-center max-w-md"
      >
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
          className="text-6xl mb-6 inline-block"
        >
          ✨
        </motion.div>
        <h3 className="text-2xl font-serif font-bold text-muse-dark mb-3">
          Crafting Your Aesthetic Alter Ego...
        </h3>
        <p className="text-gray-700 font-serif mb-6">
          Our AI is romanticizing your life and blending archetypes just for you.
        </p>
        <div className="flex justify-center gap-2">
          {[0, 1, 2].map((i) => (
            <motion.div
              key={i}
              animate={{ y: [0, -10, 0] }}
              transition={{ duration: 1, repeat: Infinity, delay: i * 0.15 }}
              className="w-3 h-3 bg-muse-rose rounded-full"
            />
          ))}
        </div>
      </motion.div>
    </motion.div>
  );
}
