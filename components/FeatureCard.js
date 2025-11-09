import React from 'react';
import { motion } from 'framer-motion';

export default function FeatureCard({ feature }) {
  return (
    <motion.div
      whileHover={{ y: -10, scale: 1.02 }}
      transition={{ duration: 0.3 }}
      className="card p-10 h-full flex flex-col items-center text-center"
    >
      <motion.div 
        className="text-6xl mb-6"
        whileHover={{ scale: 1.2, rotate: 5 }}
        transition={{ duration: 0.3 }}
      >
        {feature.icon}
      </motion.div>
      <h3 className="text-2xl font-serif font-bold text-muse-dark mb-4">
        {feature.title}
      </h3>
      <p className="text-gray-600 font-serif leading-relaxed">
        {feature.description}
      </p>
    </motion.div>
  );
}
