import React, { useState, useEffect } from 'react';
import RacingGame from './RacingGame';

function App() {
  const [currentScreen, setCurrentScreen] = useState('menu');
  const [showSettings, setShowSettings] = useState(false);
  const [leaderboard] = useState([
    { name: 'NEXUS', time: '45.2s', position: 1 },
    { name: 'CIPHER', time: '47.8s', position: 2 },
    { name: 'VECTOR', time: '49.1s', position: 3 },
    { name: 'QUANTUM', time: '51.3s', position: 4 },
    { name: 'PHOENIX', time: '53.7s', position: 5 },
  ]);

  const [settings, setSettings] = useState({
    volume: 80,
    difficulty: 'medium',
    graphics: 'high',
  });

  useEffect(() => {
    // Prevent scrolling
    document.body.style.overflow = 'hidden';
    return () => {
      document.body.style.overflow = 'auto';
    };
  }, []);

  if (currentScreen === 'game') {
    return <RacingGame onBack={() => setCurrentScreen('menu')} />;
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900 relative overflow-hidden">
      {/* Animated background */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute w-96 h-96 bg-neon-blue rounded-full blur-3xl opacity-20 animate-pulse-slow" style={{ top: '10%', left: '10%' }} />
        <div className="absolute w-96 h-96 bg-neon-purple rounded-full blur-3xl opacity-20 animate-pulse-slow" style={{ top: '60%', right: '10%', animationDelay: '1s' }} />
        <div className="absolute w-96 h-96 bg-neon-pink rounded-full blur-3xl opacity-20 animate-pulse-slow" style={{ bottom: '10%', left: '50%', animationDelay: '2s' }} />
      </div>

      {/* Scan line effect */}
      <div className="absolute inset-0 pointer-events-none opacity-10">
        <div className="absolute w-full h-1 bg-gradient-to-r from-transparent via-neon-cyan to-transparent scan-line" />
      </div>

      {/* Grid background */}
      <div className="absolute inset-0 opacity-10 pointer-events-none" style={{
        backgroundImage: 'linear-gradient(#00f3ff 1px, transparent 1px), linear-gradient(90deg, #00f3ff 1px, transparent 1px)',
        backgroundSize: '50px 50px'
      }} />

      <div className="relative z-10 min-h-screen flex items-center justify-center p-8">
        {currentScreen === 'menu' && (
          <div className="text-center space-y-8 animate-fade-in">
            {/* Title */}
            <div className="space-y-4">
              <h1 className="text-8xl font-black text-transparent bg-clip-text bg-gradient-to-r from-neon-cyan via-neon-purple to-neon-pink orbitron text-glow animate-pulse">
                4D RACER
              </h1>
              <p className="text-2xl text-neon-cyan orbitron tracking-widest">
                BEYOND DIMENSIONS
              </p>
              <div className="flex items-center justify-center space-x-2 text-gray-400">
                <div className="w-2 h-2 bg-neon-cyan rounded-full animate-pulse" />
                <span className="text-sm orbitron">SYSTEM ONLINE</span>
                <div className="w-2 h-2 bg-neon-cyan rounded-full animate-pulse" />
              </div>
            </div>

            {/* Menu buttons */}
            <div className="space-y-4 max-w-md mx-auto">
              <button
                onClick={() => setCurrentScreen('game')}
                className="w-full px-8 py-4 bg-gradient-to-r from-neon-blue to-neon-purple hover:from-neon-purple hover:to-neon-pink text-white text-2xl font-bold rounded-lg orbitron transition-all duration-300 transform hover:scale-105 neon-border shadow-lg hover:shadow-neon-cyan"
              >
                START RACE
              </button>

              <button
                onClick={() => setCurrentScreen('leaderboard')}
                className="w-full px-8 py-4 glass-effect hover:bg-opacity-80 text-neon-cyan text-xl font-bold rounded-lg orbitron transition-all duration-300 transform hover:scale-105 border border-neon-cyan hover:shadow-lg hover:shadow-neon-cyan"
              >
                LEADERBOARD
              </button>

              <button
                onClick={() => setShowSettings(true)}
                className="w-full px-8 py-4 glass-effect hover:bg-opacity-80 text-neon-purple text-xl font-bold rounded-lg orbitron transition-all duration-300 transform hover:scale-105 border border-neon-purple hover:shadow-lg hover:shadow-neon-purple"
              >
                SETTINGS
              </button>

              <button
                onClick={() => setCurrentScreen('howto')}
                className="w-full px-8 py-4 glass-effect hover:bg-opacity-80 text-yellow-400 text-xl font-bold rounded-lg orbitron transition-all duration-300 transform hover:scale-105 border border-yellow-400 hover:shadow-lg hover:shadow-yellow-400"
              >
                HOW TO PLAY
              </button>
            </div>

            {/* Footer */}
            <div className="text-gray-500 text-sm orbitron mt-12">
              <p>© 2025 4D RACER | POWERED BY QUANTUM TECH</p>
            </div>
          </div>
        )}

        {currentScreen === 'leaderboard' && (
          <div className="max-w-2xl w-full space-y-6 animate-fade-in">
            <div className="text-center space-y-2">
              <h2 className="text-6xl font-bold text-neon-cyan orbitron text-glow">
                LEADERBOARD
              </h2>
              <p className="text-gray-400 orbitron">TOP RACERS</p>
            </div>

            <div className="glass-effect rounded-2xl p-6 space-y-3">
              {leaderboard.map((entry, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between p-4 bg-black bg-opacity-30 rounded-lg border border-neon-cyan border-opacity-30 hover:border-opacity-100 transition-all"
                >
                  <div className="flex items-center space-x-4">
                    <div className={`text-3xl font-bold orbitron ${
                      index === 0 ? 'text-yellow-400' :
                      index === 1 ? 'text-gray-300' :
                      index === 2 ? 'text-orange-400' :
                      'text-neon-cyan'
                    }`}>
                      #{entry.position}
                    </div>
                    <div className="text-xl font-bold text-white orbitron">
                      {entry.name}
                    </div>
                  </div>
                  <div className="text-2xl font-bold text-neon-purple orbitron">
                    {entry.time}
                  </div>
                </div>
              ))}
            </div>

            <button
              onClick={() => setCurrentScreen('menu')}
              className="w-full px-8 py-4 glass-effect hover:bg-opacity-80 text-white text-xl font-bold rounded-lg orbitron transition-all duration-300 transform hover:scale-105 border border-white"
            >
              BACK TO MENU
            </button>
          </div>
        )}

        {currentScreen === 'howto' && (
          <div className="max-w-3xl w-full space-y-6 animate-fade-in">
            <div className="text-center space-y-2">
              <h2 className="text-6xl font-bold text-neon-cyan orbitron text-glow">
                HOW TO PLAY
              </h2>
              <p className="text-gray-400 orbitron">MASTER THE 4TH DIMENSION</p>
            </div>

            <div className="glass-effect rounded-2xl p-8 space-y-6">
              <div className="space-y-4">
                <div className="flex items-start space-x-4">
                  <div className="w-12 h-12 bg-neon-blue rounded-lg flex items-center justify-center flex-shrink-0">
                    <span className="text-2xl">🎮</span>
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-neon-cyan orbitron mb-2">CONTROLS</h3>
                    <ul className="text-gray-300 space-y-1">
                      <li>• <span className="text-white font-bold">↑ / W</span> - Accelerate</li>
                      <li>• <span className="text-white font-bold">↓ / S</span> - Brake / Reverse</li>
                      <li>• <span className="text-white font-bold">← → / A D</span> - Steer Left/Right</li>
                      <li>• <span className="text-white font-bold">ESC</span> - Pause Game</li>
                    </ul>
                  </div>
                </div>

                <div className="flex items-start space-x-4">
                  <div className="w-12 h-12 bg-neon-purple rounded-lg flex items-center justify-center flex-shrink-0">
                    <span className="text-2xl">🏁</span>
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-neon-purple orbitron mb-2">OBJECTIVE</h3>
                    <p className="text-gray-300">
                      Complete 3 laps around the track as fast as possible. Compete against 5 AI opponents and try to finish in 1st place!
                    </p>
                  </div>
                </div>

                <div className="flex items-start space-x-4">
                  <div className="w-12 h-12 bg-neon-pink rounded-lg flex items-center justify-center flex-shrink-0">
                    <span className="text-2xl">✨</span>
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-neon-pink orbitron mb-2">4D EFFECTS</h3>
                    <p className="text-gray-300">
                      Experience time distortion effects as you accelerate. The faster you go, the more the 4th dimension warps around you with dynamic lighting and particle effects.
                    </p>
                  </div>
                </div>

                <div className="flex items-start space-x-4">
                  <div className="w-12 h-12 bg-yellow-400 rounded-lg flex items-center justify-center flex-shrink-0">
                    <span className="text-2xl">💡</span>
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-yellow-400 orbitron mb-2">TIPS</h3>
                    <ul className="text-gray-300 space-y-1">
                      <li>• Stay in the center lanes for optimal speed</li>
                      <li>• Avoid hitting the barriers</li>
                      <li>• Watch your speed meter at the bottom</li>
                      <li>• Keep an eye on your position and lap count</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>

            <button
              onClick={() => setCurrentScreen('menu')}
              className="w-full px-8 py-4 glass-effect hover:bg-opacity-80 text-white text-xl font-bold rounded-lg orbitron transition-all duration-300 transform hover:scale-105 border border-white"
            >
              BACK TO MENU
            </button>
          </div>
        )}
      </div>

      {/* Settings Modal */}
      {showSettings && (
        <div className="fixed inset-0 bg-black bg-opacity-80 flex items-center justify-center z-50 p-8">
          <div className="glass-effect rounded-2xl p-8 max-w-md w-full space-y-6 animate-fade-in">
            <h2 className="text-4xl font-bold text-neon-cyan orbitron text-glow text-center">
              SETTINGS
            </h2>

            <div className="space-y-6">
              {/* Volume */}
              <div>
                <label className="block text-neon-blue orbitron mb-2">
                  VOLUME: {settings.volume}%
                </label>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={settings.volume}
                  onChange={(e) => setSettings({ ...settings, volume: e.target.value })}
                  className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-neon-cyan"
                />
              </div>

              {/* Difficulty */}
              <div>
                <label className="block text-neon-purple orbitron mb-2">
                  DIFFICULTY
                </label>
                <select
                  value={settings.difficulty}
                  onChange={(e) => setSettings({ ...settings, difficulty: e.target.value })}
                  className="w-full px-4 py-3 bg-black bg-opacity-50 border border-neon-purple rounded-lg text-white orbitron focus:outline-none focus:border-neon-cyan"
                >
                  <option value="easy">EASY</option>
                  <option value="medium">MEDIUM</option>
                  <option value="hard">HARD</option>
                  <option value="extreme">EXTREME</option>
                </select>
              </div>

              {/* Graphics */}
              <div>
                <label className="block text-neon-pink orbitron mb-2">
                  GRAPHICS QUALITY
                </label>
                <select
                  value={settings.graphics}
                  onChange={(e) => setSettings({ ...settings, graphics: e.target.value })}
                  className="w-full px-4 py-3 bg-black bg-opacity-50 border border-neon-pink rounded-lg text-white orbitron focus:outline-none focus:border-neon-cyan"
                >
                  <option value="low">LOW</option>
                  <option value="medium">MEDIUM</option>
                  <option value="high">HIGH</option>
                  <option value="ultra">ULTRA</option>
                </select>
              </div>
            </div>

            <div className="flex space-x-4">
              <button
                onClick={() => setShowSettings(false)}
                className="flex-1 px-6 py-3 bg-gradient-to-r from-neon-blue to-neon-purple hover:from-neon-purple hover:to-neon-pink text-white font-bold rounded-lg orbitron transition-all duration-300 transform hover:scale-105"
              >
                SAVE
              </button>
              <button
                onClick={() => setShowSettings(false)}
                className="flex-1 px-6 py-3 glass-effect hover:bg-opacity-80 text-white font-bold rounded-lg orbitron transition-all duration-300 border border-white"
              >
                CANCEL
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
