import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';

const RacingGame = ({ onBack }) => {
  const mountRef = useRef(null);
  const [gameState, setGameState] = useState({
    speed: 0,
    lap: 1,
    position: 1,
    time: 0,
    isPaused: false,
    isFinished: false,
  });

  useEffect(() => {
    const mountElement = mountRef.current;
    if (!mountElement) return;

    // Scene setup
    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x000510, 0.015);
    
    const camera = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );
    
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setClearColor(0x000510);
    mountElement.appendChild(renderer.domElement);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(5, 10, 5);
    scene.add(directionalLight);

    // Point lights for neon effect
    const neonLight1 = new THREE.PointLight(0x00f3ff, 2, 50);
    neonLight1.position.set(0, 5, 0);
    scene.add(neonLight1);

    const neonLight2 = new THREE.PointLight(0xb537ff, 2, 50);
    neonLight2.position.set(0, 5, 20);
    scene.add(neonLight2);

    // Create racing track with 4D effects
    const trackGroup = new THREE.Group();
    
    // Main track
    const trackGeometry = new THREE.PlaneGeometry(20, 500);
    const trackMaterial = new THREE.MeshStandardMaterial({
      color: 0x1a1a2e,
      roughness: 0.8,
      metalness: 0.2,
    });
    const track = new THREE.Mesh(trackGeometry, trackMaterial);
    track.rotation.x = -Math.PI / 2;
    trackGroup.add(track);

    // Track lines (center and sides)
    const lineGeometry = new THREE.PlaneGeometry(0.3, 500);
    const lineMaterial = new THREE.MeshBasicMaterial({ color: 0x00f3ff });
    
    for (let i = -3; i <= 3; i++) {
      if (i === 0) continue;
      const line = new THREE.Mesh(lineGeometry, lineMaterial);
      line.rotation.x = -Math.PI / 2;
      line.position.x = i * 3;
      line.position.y = 0.01;
      trackGroup.add(line);
    }

    // Dashed center line
    for (let z = -250; z < 250; z += 10) {
      const dashGeometry = new THREE.PlaneGeometry(0.5, 4);
      const dash = new THREE.Mesh(dashGeometry, new THREE.MeshBasicMaterial({ color: 0xffff00 }));
      dash.rotation.x = -Math.PI / 2;
      dash.position.y = 0.02;
      dash.position.z = z;
      trackGroup.add(dash);
    }

    // Track barriers with neon glow
    const barrierGeometry = new THREE.BoxGeometry(1, 2, 500);
    const barrierMaterial = new THREE.MeshStandardMaterial({
      color: 0x00f3ff,
      emissive: 0x00f3ff,
      emissiveIntensity: 0.5,
    });
    
    const leftBarrier = new THREE.Mesh(barrierGeometry, barrierMaterial);
    leftBarrier.position.x = -11;
    leftBarrier.position.y = 1;
    trackGroup.add(leftBarrier);

    const rightBarrier = new THREE.Mesh(barrierGeometry, barrierMaterial);
    rightBarrier.position.x = 11;
    rightBarrier.position.y = 1;
    trackGroup.add(rightBarrier);

    scene.add(trackGroup);

    // Create player vehicle
    const vehicleGroup = new THREE.Group();
    
    // Vehicle body
    const bodyGeometry = new THREE.BoxGeometry(2, 0.8, 4);
    const bodyMaterial = new THREE.MeshStandardMaterial({
      color: 0xff006e,
      metalness: 0.8,
      roughness: 0.2,
    });
    const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
    body.position.y = 0.4;
    vehicleGroup.add(body);

    // Vehicle cockpit
    const cockpitGeometry = new THREE.BoxGeometry(1.5, 0.6, 2);
    const cockpitMaterial = new THREE.MeshStandardMaterial({
      color: 0x00f3ff,
      metalness: 0.9,
      roughness: 0.1,
      transparent: true,
      opacity: 0.7,
    });
    const cockpit = new THREE.Mesh(cockpitGeometry, cockpitMaterial);
    cockpit.position.y = 1;
    cockpit.position.z = -0.5;
    vehicleGroup.add(cockpit);

    // Vehicle wheels
    const wheelGeometry = new THREE.CylinderGeometry(0.4, 0.4, 0.3, 16);
    const wheelMaterial = new THREE.MeshStandardMaterial({ color: 0x333333 });
    
    const wheelPositions = [
      [-1.2, 0.4, 1.5],
      [1.2, 0.4, 1.5],
      [-1.2, 0.4, -1.5],
      [1.2, 0.4, -1.5],
    ];

    wheelPositions.forEach(pos => {
      const wheel = new THREE.Mesh(wheelGeometry, wheelMaterial);
      wheel.rotation.z = Math.PI / 2;
      wheel.position.set(...pos);
      vehicleGroup.add(wheel);
    });

    // Vehicle trail effect
    const trailGeometry = new THREE.PlaneGeometry(2, 0.1);
    const trailMaterial = new THREE.MeshBasicMaterial({
      color: 0xff006e,
      transparent: true,
      opacity: 0.6,
    });
    const trail = new THREE.Mesh(trailGeometry, trailMaterial);
    trail.rotation.x = -Math.PI / 2;
    trail.position.y = 0.05;
    trail.position.z = 3;
    vehicleGroup.add(trail);

    vehicleGroup.position.y = 0;
    vehicleGroup.position.z = 0;
    scene.add(vehicleGroup);

    // Create AI opponents
    const opponents = [];
    for (let i = 0; i < 5; i++) {
      const opponentGroup = new THREE.Group();
      
      const opponentBody = new THREE.Mesh(
        new THREE.BoxGeometry(2, 0.8, 4),
        new THREE.MeshStandardMaterial({
          color: [0x00ff00, 0x0000ff, 0xffff00, 0xff00ff, 0x00ffff][i],
          metalness: 0.8,
          roughness: 0.2,
        })
      );
      opponentBody.position.y = 0.4;
      opponentGroup.add(opponentBody);

      const opponentCockpit = new THREE.Mesh(cockpitGeometry, cockpitMaterial);
      opponentCockpit.position.y = 1;
      opponentCockpit.position.z = -0.5;
      opponentGroup.add(opponentCockpit);

      wheelPositions.forEach(pos => {
        const wheel = new THREE.Mesh(wheelGeometry, wheelMaterial);
        wheel.rotation.z = Math.PI / 2;
        wheel.position.set(...pos);
        opponentGroup.add(wheel);
      });

      opponentGroup.position.x = (Math.random() - 0.5) * 15;
      opponentGroup.position.z = -20 - i * 15;
      opponentGroup.userData = {
        speed: 0.15 + Math.random() * 0.1,
        lane: opponentGroup.position.x,
      };
      
      scene.add(opponentGroup);
      opponents.push(opponentGroup);
    }

    // 4D effect particles
    const particlesGeometry = new THREE.BufferGeometry();
    const particlesCount = 1000;
    const positions = new Float32Array(particlesCount * 3);

    for (let i = 0; i < particlesCount * 3; i += 3) {
      positions[i] = (Math.random() - 0.5) * 50;
      positions[i + 1] = Math.random() * 20;
      positions[i + 2] = (Math.random() - 0.5) * 500;
    }

    particlesGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const particlesMaterial = new THREE.PointsMaterial({
      color: 0x00f3ff,
      size: 0.1,
      transparent: true,
      opacity: 0.6,
    });
    const particles = new THREE.Points(particlesGeometry, particlesMaterial);
    scene.add(particles);

    // Camera position
    camera.position.set(0, 5, 10);
    camera.lookAt(vehicleGroup.position);

    // Game state
    let speed = 0;
    let maxSpeed = 0.5;
    let acceleration = 0.01;
    let deceleration = 0.005;
    let trackPosition = 0;
    let lapDistance = 400;
    let currentLap = 1;
    let totalLaps = 3;
    let startTime = Date.now();
    let isPaused = false;
    let isFinished = false;

    // Controls
    const keys = {
      ArrowUp: false,
      ArrowDown: false,
      ArrowLeft: false,
      ArrowRight: false,
      KeyW: false,
      KeyS: false,
      KeyA: false,
      KeyD: false,
      Escape: false,
    };

    const handleKeyDown = (e) => {
      if (e.code in keys) {
        keys[e.code] = true;
        if (e.code === 'Escape') {
          isPaused = !isPaused;
          setGameState(prev => ({ ...prev, isPaused }));
        }
      }
    };

    const handleKeyUp = (e) => {
      if (e.code in keys) {
        keys[e.code] = false;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);

    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate);

      if (isPaused || isFinished) {
        renderer.render(scene, camera);
        return;
      }

      // Player controls
      if (keys.ArrowUp || keys.KeyW) {
        speed = Math.min(speed + acceleration, maxSpeed);
      } else if (keys.ArrowDown || keys.KeyS) {
        speed = Math.max(speed - acceleration, -maxSpeed * 0.5);
      } else {
        if (speed > 0) speed = Math.max(0, speed - deceleration);
        if (speed < 0) speed = Math.min(0, speed + deceleration);
      }

      if (keys.ArrowLeft || keys.KeyA) {
        vehicleGroup.position.x = Math.max(-9, vehicleGroup.position.x - 0.15);
      }
      if (keys.ArrowRight || keys.KeyD) {
        vehicleGroup.position.x = Math.min(9, vehicleGroup.position.x + 0.15);
      }

      // Move track
      trackPosition += speed;
      trackGroup.position.z = trackPosition % lapDistance;

      // Update lap counter
      if (trackPosition > lapDistance * currentLap) {
        currentLap++;
        if (currentLap > totalLaps) {
          isFinished = true;
          setGameState(prev => ({ ...prev, isFinished: true }));
        }
      }

      // Move opponents
      opponents.forEach((opponent, index) => {
        opponent.position.z += opponent.userData.speed;
        
        if (opponent.position.z > vehicleGroup.position.z + 30) {
          opponent.position.z = vehicleGroup.position.z - 50 - Math.random() * 50;
          opponent.position.x = (Math.random() - 0.5) * 15;
        }

        // Simple AI lane changing
        if (Math.random() < 0.01) {
          opponent.userData.lane = (Math.random() - 0.5) * 15;
        }
        opponent.position.x += (opponent.userData.lane - opponent.position.x) * 0.02;
      });

      // Animate particles (4D effect)
      const positions = particles.geometry.attributes.position.array;
      for (let i = 0; i < positions.length; i += 3) {
        positions[i + 2] += speed * 2;
        if (positions[i + 2] > 250) {
          positions[i + 2] = -250;
        }
      }
      particles.geometry.attributes.position.needsUpdate = true;

      // Rotate particles for 4D effect
      particles.rotation.y += 0.001;

      // 4D time distortion effect based on speed
      const timeFactor = 1 + speed * 2;
      neonLight1.intensity = 2 + Math.sin(Date.now() * 0.001 * timeFactor) * 0.5;
      neonLight2.intensity = 2 + Math.cos(Date.now() * 0.001 * timeFactor) * 0.5;

      // Camera follow with dynamic angle
      camera.position.z = vehicleGroup.position.z + 10 - speed * 5;
      camera.position.y = 5 + speed * 3;
      camera.lookAt(vehicleGroup.position);

      // Update game state
      const elapsedTime = (Date.now() - startTime) / 1000;
      const playerPosition = 1 + opponents.filter(opp => opp.position.z > vehicleGroup.position.z).length;
      
      setGameState({
        speed: Math.abs(speed * 200).toFixed(0),
        lap: currentLap,
        position: playerPosition,
        time: elapsedTime.toFixed(1),
        isPaused,
        isFinished,
      });

      renderer.render(scene, camera);
    };

    animate();

    // Handle window resize
    const handleResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    };
    window.addEventListener('resize', handleResize);

    // Cleanup
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
      window.removeEventListener('resize', handleResize);
      if (mountElement && renderer.domElement) {
        mountElement.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, []);

  return (
    <div className="relative w-full h-screen">
      <div ref={mountRef} className="w-full h-full" />
      
      {/* HUD */}
      <div className="absolute top-0 left-0 right-0 p-6 pointer-events-none">
        <div className="flex justify-between items-start">
          {/* Left HUD */}
          <div className="glass-effect rounded-lg p-4 space-y-2">
            <div className="text-neon-cyan text-2xl font-bold orbitron">
              SPEED: <span className="text-white">{gameState.speed}</span> KM/H
            </div>
            <div className="text-neon-blue text-xl orbitron">
              LAP: <span className="text-white">{gameState.lap}/3</span>
            </div>
            <div className="text-yellow-400 text-xl orbitron">
              POSITION: <span className="text-white">#{gameState.position}</span>
            </div>
          </div>

          {/* Right HUD */}
          <div className="glass-effect rounded-lg p-4">
            <div className="text-neon-purple text-2xl font-bold orbitron">
              TIME: <span className="text-white">{gameState.time}s</span>
            </div>
          </div>
        </div>
      </div>

      {/* Speed bar */}
      <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 w-96 pointer-events-none">
        <div className="glass-effect rounded-full h-4 overflow-hidden">
          <div 
            className="h-full bg-gradient-to-r from-neon-blue via-neon-purple to-neon-pink transition-all duration-100"
            style={{ width: `${(gameState.speed / 100) * 100}%` }}
          />
        </div>
        <div className="text-center text-neon-cyan text-sm mt-2 orbitron">VELOCITY METER</div>
      </div>

      {/* Controls hint */}
      <div className="absolute bottom-8 left-8 glass-effect rounded-lg p-3 text-sm text-gray-300 pointer-events-none">
        <div className="orbitron text-neon-cyan mb-2">CONTROLS:</div>
        <div>↑/W - Accelerate</div>
        <div>↓/S - Brake</div>
        <div>←/→ or A/D - Steer</div>
        <div>ESC - Pause</div>
      </div>

      {/* Pause menu */}
      {gameState.isPaused && (
        <div className="absolute inset-0 bg-black bg-opacity-80 flex items-center justify-center pointer-events-auto">
          <div className="glass-effect rounded-2xl p-8 text-center space-y-6">
            <h2 className="text-5xl font-bold text-neon-cyan orbitron text-glow">PAUSED</h2>
            <p className="text-xl text-gray-300">Press ESC to resume</p>
            <button
              onClick={onBack}
              className="px-8 py-3 bg-red-600 hover:bg-red-700 text-white rounded-lg orbitron font-bold transition-all transform hover:scale-105"
            >
              EXIT TO MENU
            </button>
          </div>
        </div>
      )}

      {/* Finish screen */}
      {gameState.isFinished && (
        <div className="absolute inset-0 bg-black bg-opacity-90 flex items-center justify-center pointer-events-auto">
          <div className="glass-effect rounded-2xl p-12 text-center space-y-6 max-w-md">
            <h2 className="text-6xl font-bold text-neon-cyan orbitron text-glow animate-pulse">
              RACE COMPLETE!
            </h2>
            <div className="space-y-3 text-2xl">
              <div className="text-yellow-400 orbitron">
                Final Position: <span className="text-white">#{gameState.position}</span>
              </div>
              <div className="text-neon-purple orbitron">
                Total Time: <span className="text-white">{gameState.time}s</span>
              </div>
            </div>
            <button
              onClick={onBack}
              className="px-8 py-4 bg-gradient-to-r from-neon-blue to-neon-purple hover:from-neon-purple hover:to-neon-pink text-white rounded-lg orbitron font-bold text-xl transition-all transform hover:scale-105 neon-border"
            >
              BACK TO MENU
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default RacingGame;
