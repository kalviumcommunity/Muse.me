import React, { useState } from 'react';
import Head from 'next/head';
import Link from 'next/link';
import { motion } from 'framer-motion';
import axios from 'axios';
import PersonaCard from '../components/PersonaCard';
import LoadingSpinner from '../components/LoadingSpinner';

// Use relative API routes (Next.js API routes in /pages/api)
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || '/api';

export default function Generate() {
  const [input, setInput] = useState('');
  const [aestheticPreference, setAestheticPreference] = useState('');
  const [loading, setLoading] = useState(false);
  const [persona, setPersona] = useState(null);
  const [error, setError] = useState(null);

  const aesthetics = [
    '✨ Cyberpunk Poet',
    '🌾 Cottagecore Romantic',
    '☁️ Cloudcore Dreamer',
    '📚 Dark Academia',
    '🎨 Maximalist Creator',
    '🌙 No Preference'
  ];

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!input.trim()) {
      setError('Please describe yourself or your daily routine');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(`${API_BASE_URL}/generate`, {
        user_input: input,
        aesthetic_preference: aestheticPreference && aestheticPreference !== '🌙 No Preference' ? aestheticPreference : null
      });

      setPersona(response.data);
    } catch (err) {
      console.error('Error generating persona:', err);
      setError(
        err.response?.data?.detail || 
        'Failed to generate persona. Please try again with a different input.'
      );
    } finally {
      setLoading(false);
    }
  };

  const containerVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.6 } }
  };

  return (
    <>
      <Head>
        <title>Generate Your Aesthetic Alter Ego | Muse.me</title>
        <meta name="description" content="Transform your life into an aesthetic alter ego" />
      </Head>

      <main className="min-h-screen bg-gradient-to-b from-muse-cream to-white py-12 px-4">
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          className="max-w-3xl mx-auto"
        >
          {/* Header */}
          <div className="text-center mb-16">
            <motion.h1 
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-6xl font-serif font-bold text-muse-dark mb-6"
            >
              ✨ Create Your Aesthetic Alter Ego
            </motion.h1>
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.2 }}
              className="section-divider"
            ></motion.div>
            <motion.p 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3 }}
              className="text-xl text-gray-600 font-serif max-w-2xl mx-auto"
            >
              Describe your daily life, and we'll romanticize it into something extraordinary
            </motion.p>
          </div>

          {!persona ? (
            // Form Section
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="bg-white rounded-3xl shadow-2xl p-10 md:p-12 mb-8 border border-gray-100"
            >
              <form onSubmit={handleSubmit} className="space-y-8">
                {/* Main Input */}
                <div>
                  <label className="flex items-center gap-2 text-xl font-serif font-bold text-muse-dark mb-4">
                    <span className="text-2xl">📝</span>
                    Tell us about yourself
                  </label>
                  <textarea
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Write about your daily routine, hobbies, personality, or any aspect of your life you'd like to romanticize..."
                    className="textarea-field h-48"
                  />
                  <p className="text-sm text-gray-500 mt-3 font-serif">
                    💡 Minimum 10 characters. Be descriptive for best results.
                  </p>
                </div>

                {/* Aesthetic Preference */}
                <div>
                  <label className="flex items-center gap-2 text-xl font-serif font-bold text-muse-dark mb-4">
                    <span className="text-2xl">🎨</span>
                    Aesthetic Preference (Optional)
                  </label>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                    {aesthetics.map((aesthetic) => (
                      <motion.button
                        key={aesthetic}
                        type="button"
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.98 }}
                        onClick={() => setAestheticPreference(aesthetic)}
                        className={`py-4 px-6 rounded-2xl font-serif font-semibold transition-all duration-200 text-center ${
                          aestheticPreference === aesthetic
                            ? 'bg-gradient-to-r from-muse-rose to-pink-400 text-white shadow-lg scale-105'
                            : 'bg-gray-50 text-muse-dark hover:bg-gray-100 border-2 border-gray-200'
                        }`}
                      >
                        {aesthetic}
                      </motion.button>
                    ))}
                  </div>
                </div>

                {/* Error Message */}
                {error && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="bg-red-50 border-2 border-red-200 p-5 rounded-2xl"
                  >
                    <div className="flex items-center gap-3">
                      <span className="text-2xl">⚠️</span>
                      <p className="text-red-700 font-serif font-medium">{error}</p>
                    </div>
                  </motion.div>
                )}

                {/* Submit Button */}
                <motion.button
                  type="submit"
                  disabled={loading}
                  whileHover={{ scale: loading ? 1 : 1.02 }}
                  whileTap={{ scale: loading ? 1 : 0.98 }}
                  className={`w-full py-5 rounded-2xl font-serif text-xl font-bold transition-all duration-300 flex items-center justify-center gap-3 ${
                    loading
                      ? 'btn-disabled'
                      : 'bg-gradient-to-r from-muse-rose via-pink-400 to-muse-rose text-white shadow-xl hover:shadow-2xl'
                  }`}
                >
                  {loading ? (
                    <>
                      <svg className="animate-spin h-6 w-6" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      <span>Crafting Your Aesthetic Alter Ego...</span>
                    </>
                  ) : (
                    <>
                      <span>✨</span>
                      <span>Generate My Aesthetic Alter Ego</span>
                      <span>→</span>
                    </>
                  )}
                </motion.button>
              </form>
            </motion.div>
          ) : (
            // Results Section
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.6 }}
            >
              <PersonaCard persona={persona} />

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="flex flex-col sm:flex-row gap-4 justify-center mt-12"
              >
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => {
                    setPersona(null);
                    setInput('');
                    setAestheticPreference('');
                    setError(null);
                  }}
                  className="px-10 py-4 bg-gradient-to-r from-muse-rose to-pink-400 text-white rounded-full font-serif text-lg font-bold shadow-xl hover:shadow-2xl transition-all duration-300 flex items-center justify-center gap-2"
                >
                  <span>✨</span>
                  <span>Create Another Persona</span>
                </motion.button>
                <Link href="/">
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.98 }}
                    className="px-10 py-4 bg-white text-muse-dark border-2 border-muse-rose rounded-full font-serif text-lg font-bold shadow-lg hover:shadow-xl transition-all duration-300 flex items-center justify-center gap-2"
                  >
                    <span>←</span>
                    <span>Back to Home</span>
                  </motion.button>
                </Link>
              </motion.div>
            </motion.div>
          )}

          {/* Loading Spinner */}
          {loading && <LoadingSpinner />}
        </motion.div>
      </main>
    </>
  );
}
