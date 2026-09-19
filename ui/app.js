const pages = document.querySelectorAll('.page');
const toast = document.getElementById('toast');
const question = document.getElementById('question');
const locationInput = document.getElementById('location');
const thread = document.getElementById('chat-thread');
const welcome = document.getElementById('welcome');

// Register GSAP Plugins
if (typeof gsap !== 'undefined' && typeof ScrollTrigger !== 'undefined') {
  gsap.registerPlugin(ScrollTrigger);
}

let cinematicTrigger = null;
let globalStartWaterfallLoop = null;
let globalStopWaterfallLoop = null;
let globalRenderFrameZero = null;
let isHomeNavigating = false;
let isDemoNavigating = false;
let isLoopingWaterfall = false;
let loopAnimationFrameId = null;
let currentLoopFrame = 402;
const LOOP_START = 402;
const LOOP_END = 642;
let lastFrameTime = 0;
const FRAME_INTERVAL = 1000 / 30; // ~30 FPS loop
let loopDirection = 1;
let hasTypedWorkspace = false;

// ---------------------------------------------------------------------------
// 1. GSAP ScrollTrigger for 60fps Canvas Frame Sequence Scrubbing (673 frames)
// ---------------------------------------------------------------------------
function initScrollTrigger() {
  if (typeof gsap === 'undefined' || typeof ScrollTrigger === 'undefined') return;

  const canvas = document.getElementById('hero-canvas');
  const section = document.getElementById('continuous-experience');
  if (!canvas || !section) return;

  const ctx = canvas.getContext('2d');
  const frameCount = 673;
  const currentFramePath = (index) => `frames/frame_${index.toString().padStart(4, '0')}.jpg`;

  const images = [];
  const sequence = { frame: 0 };

  const resizeCanvas = () => {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const w = Math.round(rect.width || window.innerWidth);
    const h = Math.round(rect.height || window.innerHeight);
    const newW = Math.round(w * dpr);
    const newH = Math.round(h * dpr);

    if (canvas.width !== newW || canvas.height !== newH) {
      canvas.width = newW;
      canvas.height = newH;
    }
    render();
  };

  const drawCoverImage = (img) => {
    if (!img || !img.complete || img.naturalWidth === 0) return;
    const w = canvas.width;
    const h = canvas.height;
    const imgRatio = img.naturalWidth / img.naturalHeight;
    const canvasRatio = w / h;

    let renderW, renderH, offsetX, offsetY;

    if (canvasRatio > imgRatio) {
      renderW = w;
      renderH = w / imgRatio;
      offsetX = 0;
      offsetY = (h - renderH) / 2;
    } else {
      renderH = h;
      renderW = h * imgRatio;
      offsetX = (w - renderW) / 2;
      offsetY = 0;
    }

    ctx.clearRect(0, 0, w, h);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(img, offsetX, offsetY, renderW, renderH);
  };

  const render = () => {
    const idx = Math.min(frameCount - 1, Math.max(0, Math.round(sequence.frame)));
    const img = images[idx];
    if (img && img.complete) {
      drawCoverImage(img);
    }
  };

  const drawCanvasFrame = (idx) => {
    const imageIdx = Math.min(frameCount - 1, Math.max(0, Math.round(idx)));
    const img = images[imageIdx];
    if (img && img.complete) {
      drawCoverImage(img);
    }
  };

  const startWaterfallLoop = () => {
    if (isLoopingWaterfall) return;
    isLoopingWaterfall = true;
    currentLoopFrame = LOOP_START;

    const bgVid = document.getElementById('workspace-bg-video');
    const cvs = document.getElementById('hero-canvas');
    if (bgVid) {
      bgVid.style.display = 'block';
      bgVid.play().catch(() => {});
      if (cvs) cvs.style.opacity = '0';
    }
    
    function loopStep(timestamp) {
      if (!isLoopingWaterfall) return;
      if (timestamp - lastFrameTime >= FRAME_INTERVAL) {
        lastFrameTime = timestamp;
        drawCanvasFrame(currentLoopFrame);
        
        currentLoopFrame++;
        if (currentLoopFrame > LOOP_END) {
          currentLoopFrame = LOOP_START;
        }
      }
      loopAnimationFrameId = requestAnimationFrame(loopStep);
    }
    loopAnimationFrameId = requestAnimationFrame(loopStep);
  };

  const stopWaterfallLoop = () => {
    if (!isLoopingWaterfall) return;
    isLoopingWaterfall = false;
    if (loopAnimationFrameId) {
      cancelAnimationFrame(loopAnimationFrameId);
      loopAnimationFrameId = null;
    }
    const bgVid = document.getElementById('workspace-bg-video');
    const cvs = document.getElementById('hero-canvas');
    if (bgVid) {
      bgVid.pause();
      bgVid.style.display = 'none';
      if (cvs) cvs.style.opacity = '1';
    }
  };

  globalStartWaterfallLoop = startWaterfallLoop;
  globalStopWaterfallLoop = stopWaterfallLoop;
  globalRenderFrameZero = () => {
    const bgVid = document.getElementById('workspace-bg-video');
    const cvs = document.getElementById('hero-canvas');
    if (bgVid) {
      bgVid.pause();
      bgVid.style.display = 'none';
      if (cvs) cvs.style.opacity = '1';
    }
    sequence.frame = 0;
    render();
  };

  // Preload all 673 frames safely with immediate render on load/cache
  for (let i = 0; i < frameCount; i++) {
    const img = new Image();
    images[i] = img;
    img.onload = () => {
      if (i === 0 || Math.round(sequence.frame) === i) {
        resizeCanvas();
        render();
      }
    };
    img.src = currentFramePath(i);
    if (img.complete) {
      if (i === 0 || Math.round(sequence.frame) === i) {
        resizeCanvas();
        render();
      }
    }
  }

  window.addEventListener('resize', resizeCanvas);
  resizeCanvas();

  if (cinematicTrigger) cinematicTrigger.kill();

  const timeline = gsap.timeline({
    scrollTrigger: {
      trigger: section,
      start: 'top top',
      end: '+=2400',
      pin: true,
      scrub: 0.3,
      anticipatePin: 1,
      invalidateOnRefresh: true,
      onUpdate: (self) => {
        if (isHomeNavigating) {
          stopWaterfallLoop();
          sequence.frame = 0;
          render();
          return;
        }
        if (isDemoNavigating) {
          startWaterfallLoop();
          return;
        }
        const frameIdx = Math.min(frameCount - 1, Math.max(0, Math.round(sequence.frame)));
        if (self.progress >= 0.78 || frameIdx >= LOOP_START) {
          startWaterfallLoop();
        } else {
          stopWaterfallLoop();
          render();
        }

        if (self.progress >= 0.52 && !hasTypedWorkspace) {
          hasTypedWorkspace = true;
          playGamingTypewriter();
        } else if (self.progress < 0.35 && hasTypedWorkspace) {
          hasTypedWorkspace = false;
          resetWorkspaceTypewriter();
        }

        if (self.progress >= 0.995) {
          activateWorkspace();
        }
      },
    },
  });

  cinematicTrigger = timeline.scrollTrigger;

  timeline
    // Scrub video frames from 0 to 672 continuous motion
    .to(sequence, {
      frame: frameCount - 1,
      ease: 'none',
    }, 0)

    // Phase 1: Cinematic Story Panels (0.0 to 0.25)
    .to('.opening-panel', { yPercent: -20, autoAlpha: 0, filter: 'blur(8px)', duration: 0.06 }, 0.05)
    .fromTo('.panel-2', { y: 40, autoAlpha: 0, filter: 'blur(10px)' }, { y: 0, autoAlpha: 1, filter: 'blur(0px)', duration: 0.05 }, 0.08)
    .to('.panel-2', { y: -50, autoAlpha: 0, filter: 'blur(8px)', duration: 0.05 }, 0.14)
    .fromTo('.final-panel', { y: 35, autoAlpha: 0, filter: 'blur(10px)' }, { y: 0, autoAlpha: 1, filter: 'blur(0px)', duration: 0.05 }, 0.17)
    .to('.final-panel', { autoAlpha: 0, filter: 'blur(8px)', duration: 0.05 }, 0.23)
    .to('.scroll-indicator', { autoAlpha: 0, duration: 0.04 }, 0.06)

    // Phase 2: Section 2 - Platform Capabilities Overlay (0.25 to 0.48)
    .fromTo('.afterglow', { autoAlpha: 0, y: 60, scale: 0.96 }, { autoAlpha: 1, y: 0, scale: 1, duration: 0.07 }, 0.25)
    .to('.afterglow', { autoAlpha: 0, y: -40, duration: 0.07 }, 0.44)

    // Phase 3: Section 3 - Chat Assistant Workspace Overlay (0.48 to 1.0)
    .to('.navigation.landing-nav', { autoAlpha: 0, duration: 0.05 }, 0.45)
    .fromTo('.workspace', { autoAlpha: 0, y: 80, scale: 0.95 }, { autoAlpha: 1, y: 0, scale: 1, duration: 0.08 }, 0.48);

  setTimeout(() => {
    ScrollTrigger.refresh();
  }, 150);
}

// ---------------------------------------------------------------------------
// 2. Initialize 3D Motion & Parallax
// ---------------------------------------------------------------------------
function init3DMotion() {
  if (typeof gsap === 'undefined') return;

  gsap.to('.planet', {
    rotationY: 360,
    repeat: -1,
    duration: 15,
    ease: 'none'
  });

  // Card & Button Micro-Interactions
  document.querySelectorAll('.feature-rail div, .suggestions button, .card').forEach(card => {
    card.addEventListener('mousemove', (e) => {
      const r = card.getBoundingClientRect();
      const cx = (e.clientX - r.left) / r.width - 0.5;
      const cy = (e.clientY - r.top) / r.height - 0.5;
      gsap.to(card, {
        rotationY: cx * 12,
        rotationX: -cy * 12,
        scale: 1.03,
        duration: 0.3,
        ease: 'power1.out'
      });
    });
    card.addEventListener('mouseleave', () => {
      gsap.to(card, {
        rotationY: 0,
        rotationX: 0,
        scale: 1,
        duration: 0.5,
        ease: 'power2.out'
      });
    });
  });
}

// ---------------------------------------------------------------------------
// Gaming Typewriter Effect for Workspace Welcome
// ---------------------------------------------------------------------------
function playGamingTypewriter() {
  const heading = document.getElementById('welcome-heading');
  const subText = document.querySelector('.welcome p');
  const suggestionButtons = document.querySelectorAll('.suggestions button');
  if (!heading) return;

  suggestionButtons.forEach(btn => { 
    btn.classList.remove('revealed'); 
    btn.style.visibility = 'hidden'; 
    btn.style.opacity = '0'; 
  });
  if (subText) subText.style.opacity = '0';

  const line1 = "WHAT WOULD YOU LIKE TO ";
  const line2 = "UNDERSTAND?";
  heading.innerHTML = '';

  const textNode1 = document.createTextNode('');
  const br = document.createElement('br');
  const textNode2 = document.createTextNode('');
  const cursor = document.createElement('span');
  cursor.className = 'game-cursor';
  cursor.textContent = '│';

  heading.appendChild(textNode1);
  heading.appendChild(cursor);

  let i = 0, j = 0;
  if (heading._typeTimer) clearInterval(heading._typeTimer);

  heading._typeTimer = setInterval(() => {
    if (i < line1.length) {
      textNode1.nodeValue += line1.charAt(i);
      i++;
    } else if (j === 0 && !heading.contains(br)) {
      heading.insertBefore(br, cursor);
      heading.insertBefore(textNode2, cursor);
      textNode2.nodeValue += line2.charAt(j);
      j++;
    } else if (j < line2.length) {
      textNode2.nodeValue += line2.charAt(j);
      j++;
    } else {
      clearInterval(heading._typeTimer);
      heading._typeTimer = null;
      if (cursor && cursor.parentNode) {
        cursor.parentNode.removeChild(cursor);
      }
      revealSubtitleAndBoxes(subText, suggestionButtons);
    }
  }, 35);
}

function revealSubtitleAndBoxes(subText, suggestionButtons) {
  const chatMain = document.querySelector('.chat-main');

  if (subText) {
    if (typeof gsap !== 'undefined') {
      gsap.to(subText, { opacity: 1, duration: 0.5, ease: 'power2.out' });
    } else {
      subText.style.opacity = '1';
    }
  }

  suggestionButtons.forEach(btn => { btn.style.visibility = 'visible'; });

  if (typeof gsap !== 'undefined') {
    gsap.fromTo(suggestionButtons, 
      { y: 35, opacity: 0, scale: 0.94, filter: 'blur(6px)' },
      { 
        y: 0, 
        opacity: 1, 
        scale: 1, 
        filter: 'blur(0px)',
        duration: 0.7, 
        stagger: 0.18, 
        ease: 'back.out(1.5)',
        onStart: () => {
          suggestionButtons.forEach((btn, idx) => {
            setTimeout(() => btn.classList.add('revealed'), idx * 180);
          });
        },
        onUpdate: () => {
          if (chatMain) {
            chatMain.scrollTop = chatMain.scrollHeight;
          }
        },
        onComplete: () => {
          gsap.set(suggestionButtons, { clearProps: 'transform,filter' });
          if (chatMain) {
            gsap.to(chatMain, { scrollTop: chatMain.scrollHeight, duration: 0.4, ease: 'power2.out' });
          }
        }
      }
    );
  } else {
    suggestionButtons.forEach((btn, index) => {
      setTimeout(() => {
        btn.classList.add('revealed');
        btn.style.opacity = '1';
        btn.style.visibility = 'visible';
        if (chatMain) chatMain.scrollTop = chatMain.scrollHeight;
      }, index * 180);
    });
  }
}

function resetWorkspaceTypewriter() {
  const heading = document.getElementById('welcome-heading');
  const subText = document.querySelector('.welcome p');
  const suggestionButtons = document.querySelectorAll('.suggestions button');
  if (heading && heading._typeTimer) {
    clearInterval(heading._typeTimer);
    heading._typeTimer = null;
  }
  if (subText) subText.style.opacity = '0';
  suggestionButtons.forEach(btn => {
    btn.classList.remove('revealed');
    btn.style.visibility = 'hidden';
    btn.style.opacity = '0';
  });
}

// ---------------------------------------------------------------------------
// Workspace Activation / Deactivation
// ---------------------------------------------------------------------------
let isWorkspaceLocked = false;

function activateWorkspace() {
  isWorkspaceLocked = true;
  const workspace = document.getElementById('workspace');
  if (workspace) workspace.classList.add('active-workspace');
  document.body.style.overflow = 'hidden';
  document.documentElement.style.overflow = 'hidden';
}

function deactivateWorkspace() {
  isWorkspaceLocked = false;
  const workspace = document.getElementById('workspace');
  if (workspace) workspace.classList.remove('active-workspace');
  document.body.style.overflow = '';
  document.documentElement.style.overflow = '';
}

function scrollToSection(id) {
  if (id === 'result') {
    document.getElementById('result').classList.add('active');
    return;
  }
  document.getElementById('result').classList.remove('active');
  
  if (id === 'landing') {
    deactivateWorkspace();
    window.scrollTo({ top: 0, behavior: 'smooth' });
    return;
  }

  if (!cinematicTrigger) return;

  if (id === 'workspace') {
    window.scrollTo({ top: cinematicTrigger.end, behavior: 'smooth' });
    setTimeout(() => {
      playGamingTypewriter();
      activateWorkspace();
    }, 450);
    return;
  }

  const totalScroll = cinematicTrigger.end - cinematicTrigger.start;
  let targetRatio = 0;
  if (id === 'explore') targetRatio = 0.30;

  const targetY = cinematicTrigger.start + totalScroll * targetRatio;
  window.scrollTo({ top: targetY, behavior: 'smooth' });
}

function show(id) {
  scrollToSection(id);
}

// ---------------------------------------------------------------------------
// Utility Functions
// ---------------------------------------------------------------------------
function notify(text) {
  toast.textContent = text;
  toast.classList.add('show');
  if (typeof gsap !== 'undefined') {
    gsap.fromTo(toast, { y: 30, opacity: 0, scale: 0.9 }, { y: 0, opacity: 1, scale: 1, duration: 0.4, ease: 'back.out(1.6)' });
  }
  setTimeout(() => toast.classList.remove('show'), 2200);
}

function escapeHtml(value) {
  return value.replace(/[&<>'"]/g, char => ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', "'":'\&#39;', '"':'&quot;' }[char]));
}

function formatMarkdown(text) {
  if (!text) return '';
  return text
    .replace(/^### (.*$)/gim, '<h3 style="margin:14px 0 6px; font-size:16px; font-weight:800; color:#0e473d;">$1</h3>')
    .replace(/^#### (.*$)/gim, '<h4 style="margin:12px 0 4px; font-size:13px; font-weight:700; color:#126e5c; text-transform:uppercase; letter-spacing:0.5px;">$1</h4>')
    .replace(/\*\*(.*?)\*\*/g, '<b>$1</b>')
    .replace(/^[•\-\*] (.*$)/gim, '<div style="margin:4px 0 4px 8px; display:flex; align-items:flex-start; gap:6px;"><span style="color:#10b981; font-weight:bold;">•</span><span>$1</span></div>')
    .replace(/\n\n/g, '<div style="height:10px;"></div>')
    .replace(/\n/g, '<br/>');
}

// ---------------------------------------------------------------------------
// 3. Interactive Before vs After Image Split Slider
// ---------------------------------------------------------------------------
function initBeforeAfterSliders() {
  document.querySelectorAll('.before-after-slider').forEach(slider => {
    const range = slider.querySelector('.slider-range');
    const afterImg = slider.querySelector('.after-img');
    const divider = slider.querySelector('.slider-divider');

    if (range && afterImg && divider) {
      const updateSlider = () => {
        const val = range.value;
        afterImg.style.clipPath = `polygon(${val}% 0, 100% 0, 100% 100%, ${val}% 100%)`;
        divider.style.left = `${val}%`;
      };
      range.addEventListener('input', updateSlider);
      updateSlider();
    }
  });
}

// ---------------------------------------------------------------------------
// 4. Multi-Sensor Band View Switcher
// ---------------------------------------------------------------------------
function initMultiBandSelectors() {
  document.querySelectorAll('.band-selector').forEach(selector => {
    const buttons = selector.querySelectorAll('.band-btn');
    buttons.forEach(btn => {
      btn.addEventListener('click', () => {
        buttons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        const band = btn.dataset.band;
        const container = btn.closest('.answer-card') || btn.closest('.result-details') || document;
        const targetImgs = container.querySelectorAll('.before-img, .after-img, .result-image');

        targetImgs.forEach(img => {
          if (band === 'sar') {
            gsap.to(img, { filter: 'hue-rotate(150deg) saturate(2) contrast(1.3)', duration: 0.5 });
          } else if (band === 'ndvi') {
            gsap.to(img, { filter: 'hue-rotate(270deg) saturate(2.5) contrast(1.2)', duration: 0.5 });
          } else if (band === 'gradcam') {
            gsap.to(img, { filter: 'saturate(3) contrast(1.5) hue-rotate(330deg)', duration: 0.5 });
          } else {
            gsap.to(img, { filter: 'none', duration: 0.5 });
          }
        });

        notify(`Switched satellite view: ${btn.textContent.trim()}`);
      });
    });
  });
}

// ---------------------------------------------------------------------------
// Typewriter Streaming Reveal with Markdown Formatting
// ---------------------------------------------------------------------------
function typeWriterStream(element, fullText, callback) {
  element.innerHTML = '';
  const cursor = document.createElement('span');
  cursor.className = 'typing-cursor';
  element.appendChild(cursor);

  let idx = 0;
  const interval = setInterval(() => {
    if (idx < fullText.length) {
      cursor.insertAdjacentText('beforebegin', fullText.charAt(idx));
      idx++;
      const chatMain = document.querySelector('.chat-main');
      if (chatMain) chatMain.scrollTo({ top: chatMain.scrollHeight, behavior: 'smooth' });
    } else {
      clearInterval(interval);
      if (cursor.parentNode) cursor.parentNode.removeChild(cursor);
      element.innerHTML = formatMarkdown(fullText);
      if (callback) callback();
    }
  }, 10);
}

// ---------------------------------------------------------------------------
// 5. REAL BACKEND CONNECTION & TELEMETRY SEQUENCE
// ---------------------------------------------------------------------------
let currentPresetId = null;
let attachedFiles = [];

// Initialize Benchmark Presets
function initPresets() {
  document.querySelectorAll('.preset-card').forEach(card => {
    card.addEventListener('click', () => {
      document.querySelectorAll('.preset-card').forEach(c => c.classList.remove('active'));
      card.classList.add('active');

      const pid = card.dataset.preset;
      currentPresetId = pid;
      attachedFiles = [];
      document.getElementById('attachment-status').textContent = '';

      if (pid === 'vqa') {
        question.value = "What does the land cover look like around Chennai?";
        locationInput.value = "Chennai, Tamil Nadu, India";
      } else if (pid === 'captioning') {
        question.value = "Describe this aerial image in detail.";
        locationInput.value = "Urban District";
      } else if (pid === 'grounding') {
        question.value = "Locate and highlight all the buildings in this scene.";
        locationInput.value = "Residential Sector";
      } else if (pid === 'change_vqa') {
        question.value = "Did the buildings change between the before and after images?";
        locationInput.value = "Urban Expansion Corridor";
      } else if (pid === 'fusion') {
        question.value = "Combine optical and SAR imagery to identify flooded areas in Assam during cloud cover.";
        locationInput.value = "Assam River Basin, India";
      } else if (pid === 'compound') {
        question.value = "Combine optical and radar imagery to identify flooded areas, and compare before and after images to assess changes.";
        locationInput.value = "Assam Flood Plain & Urban Periphery";
      }
      notify(`Preset loaded: ${card.querySelector('b').textContent}`);
      question.focus();
    });
  });

  // Clear active preset ID when user types a custom query or edits location
  question.addEventListener('input', () => {
    currentPresetId = null;
    document.querySelectorAll('.preset-card').forEach(c => c.classList.remove('active'));
  });
  locationInput.addEventListener('input', () => {
    currentPresetId = null;
    document.querySelectorAll('.preset-card').forEach(c => c.classList.remove('active'));
  });
}

// ---------------------------------------------------------------------------
// WOW Flight Sequence (Real Backend-Connected)
// ---------------------------------------------------------------------------
function runWowFlightSequence(place, prompt) {
  const overlay = document.getElementById('wow-overlay');
  const satImage = document.getElementById('wow-sat-image');
  const targetLabel = document.getElementById('wow-target-label');
  const statusBadge = document.getElementById('wow-status-badge');
  const coordsText = document.getElementById('wow-coords-text');
  const progressBar = document.getElementById('wow-progress-bar');
  const laserBeam = document.getElementById('wow-laser-beam');
  const liveConf = document.getElementById('wow-live-conf');
  const steps = document.querySelectorAll('.step-item');
  const nodes = document.querySelectorAll('.wow-ai-nodes .node');

  if (!overlay) return (data, cb) => { if (cb) cb(); };

  overlay.classList.add('open');
  satImage.classList.remove('zoomed');
  laserBeam.classList.remove('active');
  progressBar.style.width = '20%';
  statusBadge.textContent = 'ORBITAL ACQUISITION ACTIVE';
  targetLabel.textContent = `ACQUIRING: ${place.toUpperCase()}`;

  const lat = (11.2588 + Math.random() * 0.08).toFixed(4);
  const lon = (75.7804 + Math.random() * 0.08).toFixed(4);
  coordsText.textContent = `LAT: ${lat}° N | LON: ${lon}° E`;

  steps.forEach(s => s.classList.remove('active'));
  nodes.forEach(n => n.classList.remove('active'));
  if (steps[0]) steps[0].classList.add('active');
  if (nodes[0]) nodes[0].classList.add('active');

  setTimeout(() => {
    satImage.classList.add('zoomed');
    progressBar.style.width = '55%';
    if (steps[1]) steps[1].classList.add('active');
    if (nodes[1]) nodes[1].classList.add('active');
    statusBadge.textContent = 'FETCHING MULTI-SENSOR BANDS';
  }, 400);

  setTimeout(() => {
    laserBeam.classList.add('active');
    progressBar.style.width = '75%';
    if (steps[2]) steps[2].classList.add('active');
    if (nodes[2]) nodes[2].classList.add('active');
    statusBadge.textContent = 'NEURAL SPECIALIST INFERENCE IN PROGRESS';
  }, 900);

  // Return completion controller
  return function finishFlight(resData, onDone) {
    if (steps[3]) steps[3].classList.add('active');
    progressBar.style.width = '100%';
    
    if (resData && resData.is_compound && liveConf) {
      liveConf.textContent = 'Multi-Specialist';
    } else if (resData && resData.is_confidence_reliable && liveConf) {
      liveConf.textContent = resData.confidence_percent + '%';
    } else if (liveConf) {
      liveConf.textContent = 'Uncalibrated';
    }

    if (resData && resData.swap_occurred) {
      statusBadge.textContent = 'GPU MODEL SWAP COMPLETE';
    } else {
      statusBadge.textContent = 'ANALYSIS COMPLETE (WARM CACHE)';
    }

    setTimeout(() => {
      gsap.to(overlay, {
        opacity: 0,
        duration: 0.45,
        onComplete: () => {
          overlay.classList.remove('open');
          overlay.style.opacity = '1';
          if (onDone) onDone();
        }
      });
    }, 450);
  };
}

// ---------------------------------------------------------------------------
// Claude-Style Conversational Response Generator (100% Dynamic from Response)
// ---------------------------------------------------------------------------
function buildClaudeChatText(prompt, place, data) {
  const safePlace = escapeHtml(place);
  const confPercent = (typeof data.confidence_percent === 'number')
    ? data.confidence_percent
    : Math.round((data.confidence || 0) * 100);
  const isReliable = Boolean(data.is_confidence_reliable);

  // Extract raw content from results or answer
  let rawContent = "";
  if (data.results && data.results.length > 0 && data.results[0].answer) {
    rawContent = data.results[0].answer;
  } else if (data.answer) {
    rawContent = data.answer;
  }

  // 1. Extract executive takeaway / plain words dynamically
  let directParagraph = "";
  const simpleWordsHtmlMatch = rawContent.match(/class="simple-words-box"[^>]*>[\s\S]*?<p>([^<]+)<\/p>/i);
  const simpleWordsMdMatch = rawContent.match(/####\s*💡\s*In Simple Words[^\n]*\n+([^\n#]+)/i);
  const assessmentMdMatch = rawContent.match(/####\s*📋\s*Analytical Assessment[^\n]*\n+([^\n#]+)/i);

  if (simpleWordsHtmlMatch && simpleWordsHtmlMatch[1]) {
    directParagraph = simpleWordsHtmlMatch[1].trim();
  } else if (simpleWordsMdMatch && simpleWordsMdMatch[1]) {
    directParagraph = simpleWordsMdMatch[1].trim();
  } else if (assessmentMdMatch && assessmentMdMatch[1]) {
    directParagraph = assessmentMdMatch[1].trim();
  } else {
    const cleaned = rawContent
      .replace(/<[^>]*>/g, ' ')
      .replace(/^#+.*$/gm, '')
      .replace(/\s+/g, ' ')
      .trim();
    const sentences = cleaned.split(/(?<=[.!?])\s+/);
    directParagraph = sentences.slice(0, 3).join(' ');
  }

  // 2. Extract primary finding / core model thesis dynamically
  let primaryFinding = "";
  if (assessmentMdMatch && assessmentMdMatch[1]) {
    primaryFinding = assessmentMdMatch[1].trim();
  } else if (data.results && data.results[0] && data.results[0].answer) {
    const lines = data.results[0].answer.split('\n').map(l => l.trim()).filter(l => l && !l.startsWith('#'));
    primaryFinding = lines[0] || "";
  }
  if (!primaryFinding) {
    primaryFinding = directParagraph;
  }

  // 3. Dynamic Confidence Badge & Text (Preserving Calibrated vs Certainty)
  let confBadgeText = "";
  if (isReliable) {
    if (data.confidence_percent < 50 && data.top_classes && data.top_classes.length >= 2) {
      const top2Str = data.top_classes.slice(0, 2).map(c => `${c[0]} (${(c[1]*100).toFixed(1)}%)`).join(' and ');
      confBadgeText = `<span style="color:#85efd0; font-weight:700;">✓ Calibrated Sub-Threshold Signals (${escapeHtml(top2Str)})</span>`;
    } else {
      confBadgeText = `<span style="color:#85efd0; font-weight:700;">✓ ${confPercent}% Calibrated Sigmoid Confidence</span>`;
    }
  } else {
    confBadgeText = `<span style="color:#fbbf24; font-weight:700;">⚠️ ~${confPercent}% Model Token Certainty</span> <em style="color:#a9c6c3; font-size:12px;">(Uncalibrated greedy decoding)</em>`;
  }

  const confNoteText = escapeHtml(data.confidence_note || (isReliable ? 'Calibrated multi-label probability from Optical-SAR dual-branch CNN' : 'Greedy decoding token certainty; does not reflect factual accuracy'));

  // 4. Dynamic Visual Evidence Description
  let evidenceDesc = "Multi-spectral optical satellite capture";
  if (data.evidence_maps) {
    if (data.evidence_maps.grounding && data.evidence_maps.grounding.boxes) {
      const bCount = data.evidence_maps.grounding.boxes.length;
      evidenceDesc = `${bCount} localized structure bounding box instance(s) with spatial coordinates`;
    } else {
      let hasHeatmap = false;
      for (const k of Object.keys(data.evidence_maps)) {
        if (data.evidence_maps[k] && data.evidence_maps[k].type === 'heatmap') {
          hasHeatmap = true;
          break;
        }
      }
      if (hasHeatmap) {
        evidenceDesc = "Co-registered Sentinel-1 SAR & Sentinel-2 Optical with Grad-CAM neural class activation heatmap";
      }
    }
  } else if (data.task_type === 'change_vqa' || (data.task_sequence && data.task_sequence.includes('change_vqa'))) {
    evidenceDesc = "Co-registered bitemporal satellite observation pair with interactive before/after delta slider";
  }

  // 5. Dynamic Specialist & Pipeline Details
  const specialistDisplay = escapeHtml(
    data.specialist_sequence && data.specialist_sequence.length > 0
      ? data.specialist_sequence.join(' → ')
      : (data.specialist || 'SatQuery Neural Specialist')
  );
  const taskDisplay = escapeHtml(
    data.task_sequence && data.task_sequence.length > 0
      ? data.task_sequence.join(' → ')
      : (data.task_type || 'Geospatial Intelligence')
  );

  // 6. Assemble dynamic key telemetry bullets
  const keyBullets = `
    <ul style="margin:8px 0; padding-left:20px; line-height:1.6; font-size:13px; color:#c5dedb;">
      <li><strong>Specialist Intelligence:</strong> ${specialistDisplay} <span style="color:#85efd0;">[${taskDisplay}]</span></li>
      <li><strong>Confidence Calibration:</strong> ${confBadgeText} — ${confNoteText}</li>
      <li><strong>Primary Finding:</strong> ${escapeHtml(primaryFinding)}</li>
      <li><strong>Visual Evidence:</strong> ${escapeHtml(evidenceDesc)}</li>
      ${data.location_telemetry && data.location_telemetry.sensor ? `<li><strong>Sensor Feed:</strong> ${escapeHtml(data.location_telemetry.sensor)}${data.location_telemetry.resolution ? ` (${escapeHtml(data.location_telemetry.resolution)} GSD)` : ''}</li>` : ''}
    </ul>
  `;

  return `
    <div class="claude-chat-body">
      <p>I have analyzed the satellite telemetry for <strong>${safePlace}</strong> regarding your query: <em>"${escapeHtml(prompt)}"</em>.</p>
      <div class="claude-takeaway">
        ${escapeHtml(directParagraph || 'Satellite analysis completed with verified telemetry and neural model inference.')}
      </div>
      <p style="margin-top:10px;"><strong>Key Telemetry Observations:</strong></p>
      ${keyBullets}
      <p style="font-size:13px; color:#85efd0; margin-top:8px;">
        💡 <em>The complete interactive evidence studio, multi-sensor comparison visualizer, and detailed geospatial intelligence report have opened in the dossier panel on the right.</em>
      </p>
    </div>
  `;
}

// ---------------------------------------------------------------------------
// Artifact Side Panel Helpers
// ---------------------------------------------------------------------------
function openArtifactPane() {
  const artifactPane = document.getElementById('artifact-pane');
  const toggleBtn = document.getElementById('toggle-artifact-pane-btn');
  if (artifactPane) {
    artifactPane.classList.add('open');
    artifactPane.setAttribute('aria-hidden', 'false');
  }
  if (toggleBtn) {
    toggleBtn.style.display = 'inline-flex';
  }
}

function closeArtifactPane() {
  const artifactPane = document.getElementById('artifact-pane');
  if (artifactPane) {
    artifactPane.classList.remove('open');
    artifactPane.classList.remove('fullscreen');
    artifactPane.setAttribute('aria-hidden', 'true');
  }
}

function toggleArtifactFullscreen() {
  const artifactPane = document.getElementById('artifact-pane');
  if (artifactPane) {
    artifactPane.classList.toggle('fullscreen');
  }
}

// ---------------------------------------------------------------------------
// Populate Artifact Side Panel with Interactive Visual Evidence & Report
// ---------------------------------------------------------------------------
function populateArtifactPane(prompt, place, data, optUrl, sarUrl) {
  const artifactPane = document.getElementById('artifact-pane');
  const artifactTitle = document.getElementById('artifact-title');
  const artifactBadge = document.getElementById('artifact-badge');
  const artifactBody = document.getElementById('artifact-body');
  if (!artifactPane || !artifactBody) return;

  const safePlace = escapeHtml(place);
  const taskLabel = (data.task_type || 'Geospatial Analysis').toUpperCase().replace(/_/g, ' ');

  if (artifactTitle) artifactTitle.textContent = `Target AOI: ${safePlace}`;
  if (artifactBadge) artifactBadge.textContent = `EARTH INTELLIGENCE · ${taskLabel}`;

  // 1. Confidence Score HTML
  let confHtml = '';
  if (data.is_compound && data.results && data.results.length > 1) {
    confHtml = '<div class="compound-confidence-box" style="display:flex; flex-direction:column; gap:8px; margin-bottom:14px;">';
    data.results.forEach(res => {
      const badgeClass = res.is_confidence_reliable ? 'calibrated' : 'uncalibrated';
      const scoreText = res.is_confidence_reliable ? `<b>${res.confidence_percent}%</b>` : '';
      confHtml += `
        <div class="score-container" style="background:rgba(3,20,24,0.6); padding:10px 14px; border-radius:8px; border:1px solid rgba(134,241,207,0.25);">
          <div style="font-size:11px; font-weight:800; color:#85efd0; margin-bottom:4px; text-transform:uppercase; letter-spacing:0.5px;">
            Step ${res.step}: ${escapeHtml(res.header_name || res.specialist)}
          </div>
          <div class="score-badge ${badgeClass}">
            <span>${escapeHtml(res.confidence_label)}</span> ${scoreText}
          </div>
          <div class="score-detail" style="margin-top:4px; color:#a9c6c3; font-size:11px;">${escapeHtml(res.confidence_note)}</div>
        </div>
      `;
    });
    confHtml += '</div>';
  } else {
    if (data.is_confidence_reliable) {
      const isSub = data.confidence_percent < 50 && data.top_classes && data.top_classes.length >= 2;
      const top2Str = isSub ? data.top_classes.slice(0, 2).map(c => `${c[0]} (${(c[1]*100).toFixed(1)}%)`).join(' and ') : '';
      const badgeHtml = isSub
        ? `<span>✓ Calibrated Sub-Threshold Signals: ${escapeHtml(top2Str)}</span>`
        : `<span>✓ Calibrated Confidence</span> <b>${data.confidence_percent}%</b>`;
      confHtml = `
        <div class="score-container" style="margin-bottom:14px;">
          <div class="score-badge calibrated">${badgeHtml}</div>
          <div class="score-detail" style="color:#a9c6c3; margin-top:4px;">${escapeHtml(data.confidence_note || 'Calibrated multi-label probability')}</div>
        </div>
      `;
    } else {
      confHtml = `
        <div class="score-container" style="margin-bottom:14px;">
          <div class="score-badge uncalibrated"><span>Model Certainty (Not Accuracy-Calibrated)</span></div>
          <div class="score-detail" style="color:#a9c6c3; margin-top:4px;">${escapeHtml(data.confidence_note || 'Greedy token probability saturated (~100%); does not reflect factual accuracy')}</div>
        </div>
      `;
    }
  }

  // 2. Evidence Maps Dynamic HTML
  let visualContentHtml = '';
  const previews = data.previews || {};

  if (data.is_compound) {
    let heatmapDataUrl = '';
    for (const r of data.results) {
      if (r.evidence_maps) {
        for (const [cls, ev] of Object.entries(r.evidence_maps)) {
          if (ev.type === 'heatmap') {
            heatmapDataUrl = ev.data_url;
            break;
          }
        }
      }
    }

    const beforeImg = previews.before || optUrl;
    const afterImg = previews.after || previews.image || optUrl;

    visualContentHtml = `
      <div class="compound-visual-section" style="margin-bottom:18px;">
        <h4 style="margin:0 0 8px; font-size:11.5px; color:#85efd0; text-transform:uppercase; letter-spacing:0.8px;">1. Optical-SAR Dual-Sensor Telemetry & Activation</h4>
        <div class="band-selector" style="margin-bottom:10px;">
          <button class="band-btn active" data-view="optical">📡 Sentinel-2 Optical</button>
          ${sarUrl ? `<button class="band-btn" data-view="sar">⚡ Sentinel-1 SAR</button>` : ''}
          ${heatmapDataUrl ? `<button class="band-btn gradcam-btn" data-view="gradcam">🔥 Grad-CAM Heatmap</button>` : ''}
        </div>
        <div class="evidence-container">
          <div class="evidence-img-wrapper" style="position:relative;">
            <img class="evidence-img main-sensor-img" src="${optUrl}" alt="Optical Observation" />
            ${heatmapDataUrl ? `<img class="gradcam-overlay-img" src="${heatmapDataUrl}" style="display:none;" alt="Grad-CAM Activation" />` : ''}
          </div>
        </div>
        <div class="legend" style="margin-top:8px;"><i></i> <span>Spatial Activation: Optical-SAR Multi-Sensor Localization</span></div>
      </div>

      <div class="compound-visual-section" style="margin-top:16px;">
        <h4 style="margin:0 0 8px; font-size:11.5px; color:#85efd0; text-transform:uppercase; letter-spacing:0.8px;">2. Bi-Temporal Change Detection Telemetry</h4>
        <div class="before-after-slider">
          <div class="slider-img before-img" style="background-image:linear-gradient(135deg,rgba(0,0,0,0.1),transparent),url('${beforeImg}');"><span>Before Image</span></div>
          <div class="slider-img after-img" style="background-image:linear-gradient(135deg,rgba(0,0,0,0.1),transparent),url('${afterImg}');"><span>After Image</span></div>
          <div class="slider-divider"><div class="slider-handle">↔</div></div>
          <input type="range" class="slider-range" min="0" max="100" value="50" aria-label="Compare satellite images" />
        </div>
        <div class="legend" style="margin-top:8px;"><i></i> <span>Bi-Temporal Comparison: Drag Slider to Compare Pre vs Post Changes</span></div>
      </div>
    `;
  } else if (data.task_type === 'optical_sar_fusion' || data.task_type === 'fusion_analysis') {
    let heatmapDataUrl = '';
    let heatmapClass = '';
    if (data.evidence_maps) {
      for (const [cls, ev] of Object.entries(data.evidence_maps)) {
        if (ev.type === 'heatmap') {
          heatmapDataUrl = ev.data_url;
          heatmapClass = cls;
          break;
        }
      }
    }

    const isSubThresholdHeatmap = data.confidence_percent < 50;
    const gradcamBtnLabel = isSubThresholdHeatmap ? '🔥 Grad-CAM (Weak Signal &lt;50%)' : '🔥 Grad-CAM Heatmap';
    const gradcamImgClass = isSubThresholdHeatmap ? 'gradcam-overlay-img sub-threshold' : 'gradcam-overlay-img';
    const subBadgeHtml = isSubThresholdHeatmap ? `<div class="subthreshold-heatmap-pill" id="subthreshold-pill" style="display:none;">⚠️ Sub-Threshold Neural Signal (&lt;50%)</div>` : '';
    const legendText = isSubThresholdHeatmap
      ? `Spatial Activation: Optical-SAR Dual-CNN Class Localization (Sub-Threshold Feature Signal: ${escapeHtml(heatmapClass || 'Candidate')})`
      : `Spatial Activation: Optical-SAR Dual-CNN Class Localization (${escapeHtml(heatmapClass || 'Active Detection')})`;

    visualContentHtml = `
      <div class="band-selector" style="margin-bottom:12px;">
        <button class="band-btn active" data-view="optical">📡 Sentinel-2 Optical</button>
        ${sarUrl ? `<button class="band-btn" data-view="sar">⚡ Sentinel-1 SAR</button>` : ''}
        ${heatmapDataUrl ? `<button class="band-btn gradcam-btn" data-view="gradcam">${gradcamBtnLabel}</button>` : ''}
      </div>
      <div class="evidence-container">
        <div class="evidence-img-wrapper" style="position:relative;">
          <img class="evidence-img main-sensor-img" src="${optUrl}" alt="Optical Observation" />
          ${heatmapDataUrl ? `<img class="${gradcamImgClass}" src="${heatmapDataUrl}" style="display:none;" alt="Grad-CAM Activation" />` : ''}
          ${subBadgeHtml}
        </div>
      </div>
      <div class="legend" style="margin-top:8px;"><i></i> <span>${legendText}</span></div>
    `;
  } else if (data.task_type === 'change_vqa') {
    const beforeImg = previews.before || optUrl;
    const afterImg = previews.after || previews.image || optUrl;

    visualContentHtml = `
      <div class="before-after-slider">
        <div class="slider-img before-img" style="background-image:linear-gradient(135deg,rgba(0,0,0,0.1),transparent),url('${beforeImg}');"><span>Before Image</span></div>
        <div class="slider-img after-img" style="background-image:linear-gradient(135deg,rgba(0,0,0,0.1),transparent),url('${afterImg}');"><span>After Image</span></div>
        <div class="slider-divider"><div class="slider-handle">↔</div></div>
        <input type="range" class="slider-range" min="0" max="100" value="50" aria-label="Compare satellite images" />
      </div>
      <div class="legend" style="margin-top:8px;"><i></i> <span>Bi-Temporal Satellite Comparison (Drag Slider)</span></div>
    `;
  } else if (data.task_type === 'grounding' && data.evidence_maps && data.evidence_maps.grounding) {
    const boxes = data.evidence_maps.grounding.boxes || [];
    const labels = data.evidence_maps.grounding.labels || [];
    let svgBoxes = '';

    boxes.forEach((b, idx) => {
      let ymin = b[0], xmin = b[1], ymax = b[2], xmax = b[3];
      if (ymax > 1.0 || xmax > 1.0) {
        ymin /= 1000; xmin /= 1000; ymax /= 1000; xmax /= 1000;
      }
      const x = (xmin * 100).toFixed(2);
      const y = (ymin * 100).toFixed(2);
      const w = Math.max(0.5, (xmax - xmin) * 100).toFixed(2);
      const h = Math.max(0.5, (ymax - ymin) * 100).toFixed(2);
      const lbl = labels[idx] || 'Building';

      svgBoxes += `
        <g>
          <rect class="bbox-rect" x="${x}%" y="${y}%" width="${w}%" height="${h}%"></rect>
          <rect class="bbox-bg" x="${x}%" y="${Math.max(0, y - 4)}%" width="${lbl.length * 7 + 10}px" height="15px"></rect>
          <text class="bbox-label" x="${parseFloat(x) + 1}%" y="${Math.max(3, y - 1)}%">${escapeHtml(lbl)}</text>
        </g>
      `;
    });

    visualContentHtml = `
      <div class="evidence-container">
        <div class="evidence-img-wrapper" style="position:relative;">
          <img class="evidence-img" src="${optUrl}" alt="Grounding Scene" />
          <svg class="bounding-box-svg" viewBox="0 0 100 100" preserveAspectRatio="none" style="position:absolute;inset:0;width:100%;height:100%;">
            ${svgBoxes}
          </svg>
        </div>
      </div>
      <div class="legend" style="margin-top:8px;"><i></i> <span>Detected <b>${boxes.length}</b> verified structure instance(s)</span></div>
    `;
  } else {
    if (optUrl) {
      visualContentHtml = `
        <div class="evidence-container">
          <div class="evidence-img-wrapper">
            <img class="evidence-img" src="${optUrl}" alt="Analyzed Satellite Scene" />
          </div>
        </div>
      `;
    }
  }

  const swapText = data.swap_occurred
    ? '<span class="swap-alert">GPU Weight Swap Performed</span>'
    : '<span style="color:#85efd0; font-weight:700;">⚡ Warm Resident (0.0s swap)</span>';

  // 3. Assemble complete artifact body HTML
  artifactBody.innerHTML = `
    <!-- 1. Visual Evidence Studio -->
    <div class="artifact-studio-section" style="background:rgba(3,20,24,0.6); border:1px solid rgba(134,241,207,0.2); border-radius:12px; padding:16px;">
      ${visualContentHtml}
    </div>

    <!-- 2. Honest Confidence Score -->
    ${confHtml}

    <!-- 3. Structured Geospatial Intelligence Report -->
    <div class="artifact-report-section" style="background:rgba(3,20,24,0.6); border:1px solid rgba(134,241,207,0.2); border-radius:12px; padding:18px;">
      ${data.answer_html || `<div class="report-card"><p>${escapeHtml(data.answer)}</p></div>`}
    </div>

    <!-- 4. Telemetry & Model Specifications -->
    <div class="artifact-telemetry-section" style="background:rgba(3,20,24,0.6); border:1px solid rgba(134,241,207,0.2); border-radius:12px; padding:16px;">
      <h4 style="margin:0 0 12px; font-size:12px; color:#85efd0; text-transform:uppercase; letter-spacing:0.8px;">🛰️ Technical Execution Dossier</h4>
      <dl style="margin:0; font-size:12px; display:grid; gap:8px;">
        <div style="display:flex; justify-content:space-between; border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:6px;"><dt style="color:#9bb8b5;">Specialist Backbone</dt><dd style="color:#fff; font-weight:600; margin:0;">${escapeHtml(data.specialist_sequence ? data.specialist_sequence.join(' → ') : (data.specialist || 'Specialist'))}</dd></div>
        <div style="display:flex; justify-content:space-between; border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:6px;"><dt style="color:#9bb8b5;">Task Sequence</dt><dd style="color:#fff; font-weight:600; margin:0;">${escapeHtml(data.task_sequence ? data.task_sequence.join(' → ') : (data.task_type || 'Analysis'))}</dd></div>
        <div style="display:flex; justify-content:space-between; border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:6px;"><dt style="color:#9bb8b5;">Inference Latency</dt><dd style="color:#fff; font-weight:600; margin:0;">${data.execution_time_s}s · ${swapText}</dd></div>
        <div style="display:flex; justify-content:space-between; border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:6px;"><dt style="color:#9bb8b5;">Target AOI</dt><dd style="color:#fff; font-weight:600; margin:0;">${safePlace}</dd></div>
        ${data.location_telemetry && data.location_telemetry.sensor ? `<div style="display:flex; justify-content:space-between; border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:6px;"><dt style="color:#9bb8b5;">Sensor Feed</dt><dd style="color:#fff; font-weight:600; margin:0;">${escapeHtml(data.location_telemetry.sensor)}</dd></div>` : ''}
        ${data.location_telemetry && data.location_telemetry.lat ? `<div style="display:flex; justify-content:space-between;"><dt style="color:#9bb8b5;">Coordinates</dt><dd style="color:#fff; font-weight:600; margin:0;">${data.location_telemetry.lat}°N, ${data.location_telemetry.lon}°E</dd></div>` : ''}
      </dl>
    </div>
  `;

  // Initialize sliders & band switchers inside artifact body
  initBeforeAfterSliders();

  const bandBtns = artifactBody.querySelectorAll('.band-btn');
  const mainImg = artifactBody.querySelector('.main-sensor-img');
  const gradcamImg = artifactBody.querySelector('.gradcam-overlay-img');

  bandBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      bandBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const view = btn.dataset.view;

      const pill = artifactBody.querySelector('#subthreshold-pill');
      if (view === 'optical') {
        if (mainImg) mainImg.src = optUrl;
        if (gradcamImg) gradcamImg.style.display = 'none';
        if (pill) pill.style.display = 'none';
      } else if (view === 'sar') {
        if (mainImg && sarUrl) mainImg.src = sarUrl;
        if (gradcamImg) gradcamImg.style.display = 'none';
        if (pill) pill.style.display = 'none';
      } else if (view === 'gradcam') {
        if (mainImg) mainImg.src = optUrl;
        if (gradcamImg) gradcamImg.style.display = 'block';
        if (pill) pill.style.display = 'flex';
      }
    });
  });

  // Wire header actions
  const copyBtn = document.getElementById('artifact-copy-btn');
  if (copyBtn) {
    copyBtn.onclick = () => {
      const textToCopy = `${data.query || prompt}\nLocation: ${place}\n\n${data.answer || ''}`;
      navigator.clipboard.writeText(textToCopy).then(() => {
        notify('Intelligence dossier copied to clipboard.');
      }).catch(() => {
        notify('Copied assessment text.');
      });
    };
  }

  const downloadBtn = document.getElementById('artifact-download-btn');
  if (downloadBtn) {
    downloadBtn.onclick = () => {
      const blob = new Blob([
        `# SatQuery Geospatial Intelligence Dossier\n\nTarget AOI: ${place}\nQuery: ${prompt}\nDate: ${new Date().toISOString()}\nSpecialist: ${data.specialist || 'SatQuery AI'}\nConfidence: ${data.confidence_percent || 90}%\n\n## Assessment\n\n${data.answer || ''}\n`
      ], { type: 'text/markdown' });
      const dlUrl = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = dlUrl;
      a.download = `SatQuery_Report_${(place || 'AOI').replace(/[^a-zA-Z0-9]/g, '_')}.md`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(dlUrl);
      notify('Dossier downloaded as Markdown.');
    };
  }
}

// ---------------------------------------------------------------------------
// Render Real AI Result (Claude-Style Chat + Right-Side Artifact Dossier)
// ---------------------------------------------------------------------------
function renderAiResult(aiMsg, prompt, place, data) {
  const aiTextTarget = aiMsg.querySelector('.ai-text-target');
  const chatArtifactSlot = aiMsg.querySelector('.chat-artifact-slot');
  const statusBadge = aiMsg.querySelector('.processing');

  const safePlace = escapeHtml(place);
  const previews = data.previews || {};
  const optUrl = previews.optical || previews.before || previews.image || 'satellite-hero.png';
  const sarUrl = previews.sar || previews.after || '';

  // 1. Build Claude-style conversational response for the chat area
  const claudeTextHtml = buildClaudeChatText(prompt, place, data);

  // 2. Build Claude Artifact Pill Card for the chat message
  const reportDossierTitle = `${safePlace} Geospatial Assessment`;
  const artifactPillHtml = `
    <div class="claude-artifact-card" data-action="open-artifact">
      <div class="artifact-card-left">
        <div class="artifact-card-icon">📑</div>
        <div class="artifact-card-text">
          <span class="artifact-card-kind">Satellite Intelligence Report</span>
          <span class="artifact-card-title">${reportDossierTitle}</span>
        </div>
      </div>
      <div class="artifact-card-right">
        <span class="artifact-pill-status">Open Dossier</span>
        <span class="artifact-pill-arrow">›</span>
      </div>
    </div>
  `;

  // Render text into chat message with smooth entrance
  aiTextTarget.innerHTML = claudeTextHtml;
  if (chatArtifactSlot) {
    chatArtifactSlot.innerHTML = artifactPillHtml;
    const cardBtn = chatArtifactSlot.querySelector('.claude-artifact-card');
    if (cardBtn) {
      cardBtn.addEventListener('click', () => openArtifactPane());
    }
  }

  // 3. Populate and open the Right-Side Artifact Dossier Panel
  populateArtifactPane(prompt, place, data, optUrl, sarUrl);
  openArtifactPane();

  if (statusBadge) {
    statusBadge.classList.remove('loading-telemetry');
    statusBadge.textContent = `Analysis complete (${data.execution_time_s}s)`;
  }

  if (typeof gsap !== 'undefined') {
    gsap.fromTo(aiTextTarget, { opacity: 0, y: 8 }, { opacity: 1, y: 0, duration: 0.35, ease: 'power2.out' });
    if (chatArtifactSlot) {
      gsap.fromTo(chatArtifactSlot, { scale: 0.96, opacity: 0, y: 10 }, { scale: 1, opacity: 1, y: 0, duration: 0.45, ease: 'back.out(1.2)' });
    }
  }

  const chatMain = document.querySelector('.chat-main');
  if (chatMain) chatMain.scrollTo({ top: chatMain.scrollHeight, behavior: 'smooth' });
}

// ---------------------------------------------------------------------------
// Real submitChat with Backend fetch('/api/query')
// ---------------------------------------------------------------------------
async function submitChat() {
  const prompt = question.value.trim();
  const place = locationInput.value.trim() || "Global Satellite Telemetry";

  if (!prompt) return notify('Write a question or select a benchmark preset above.');

  welcome.hidden = true;
  activateWorkspace();

  const safePrompt = escapeHtml(prompt);
  const safePlace = escapeHtml(place);

  // 1. Append User Message
  const userHtml = `<article class="message user-message"><div class="message-label">You <span>${safePlace}</span></div><p>${safePrompt}</p></article>`;
  thread.insertAdjacentHTML('beforeend', userHtml);

  // 2. Append AI Loading Placeholder (Claude-Style)
  const aiHtml = `
    <article class="message ai-message" id="current-ai-message">
      <div class="message-label">
        <i>◒</i> SatQueryAI 
        <span class="processing loading-telemetry"><span class="pulse-spinner"></span> 📡 Processing Telemetry Stream...</span>
      </div>
      <div class="ai-text-target claude-chat-body"></div>
      <div class="chat-artifact-slot"></div>
    </article>
  `;
  thread.insertAdjacentHTML('beforeend', aiHtml);
  const aiMsg = thread.lastElementChild;

  const chatMain = document.querySelector('.chat-main');
  if (chatMain) chatMain.scrollTo({ top: chatMain.scrollHeight, behavior: 'smooth' });

  // 3. Trigger HUD sequence
  const finishFlight = runWowFlightSequence(place, prompt);

  // 4. Prepare Payload & Execute Fetch
  try {
    const formData = new FormData();
    formData.append('query', prompt);
    formData.append('location', place);

    const activePresetId = currentPresetId;
    currentPresetId = null; // Clear immediately so subsequent messages don't retain old preset

    if (activePresetId) {
      formData.append('preset_id', activePresetId);
    } else if (attachedFiles.length === 1) {
      formData.append('image', attachedFiles[0]);
    } else if (attachedFiles.length === 4) {
      formData.append('optical', attachedFiles[0]);
      formData.append('sar', attachedFiles[1]);
      formData.append('before', attachedFiles[2]);
      formData.append('after', attachedFiles[3]);
    } else if (attachedFiles.length >= 2) {
      const pLower = prompt.toLowerCase();
      if (pLower.includes('before') || pLower.includes('change')) {
        formData.append('before', attachedFiles[0]);
        formData.append('after', attachedFiles[1]);
      } else {
        formData.append('optical', attachedFiles[0]);
        formData.append('sar', attachedFiles[1]);
      }
    }

    const response = await fetch('/api/query', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errJson = await response.json().catch(() => ({}));
      throw new Error(errJson.error || `Server error (${response.status})`);
    }

    const resData = await response.json();

    finishFlight(resData, () => {
      renderAiResult(aiMsg, prompt, place, resData);
    });

  } catch (err) {
    console.error('Query execution error:', err);
    finishFlight(null, () => {
      const statusBadge = aiMsg.querySelector('.processing');
      if (statusBadge) {
        statusBadge.textContent = 'Execution Failed';
        statusBadge.style.background = '#fdebea';
        statusBadge.style.color = '#b91c1c';
      }
      const aiTextTarget = aiMsg.querySelector('.ai-text-target');
      aiTextTarget.innerHTML = `<span style="color:#dc2626; font-weight:600;">Error processing query: ${escapeHtml(err.message)}</span>`;
    });
  }

  question.value = '';
  attachedFiles = [];
  document.getElementById('attachment-status').textContent = '';
}

// ---------------------------------------------------------------------------
// AUTHENTICATION & USER ACCOUNT STATE MANAGEMENT
// ---------------------------------------------------------------------------
let currentUser = null;
let authToken = localStorage.getItem('satquery_token') || 'demo_token_priya';

async function initAuth() {
  try {
    const res = await fetch('/api/auth/user', {
      headers: {
        'Authorization': `Bearer ${authToken}`
      }
    });
    if (res.ok) {
      const data = await res.json();
      if (data.success && data.user) {
        currentUser = data.user;
        updateUIUser(currentUser);
        return;
      }
    }
  } catch (err) {
    console.warn('Auth fetch error, using local fallback:', err);
  }

  // Fallback default persona
  currentUser = {
    id: 'priya',
    name: 'Priya Menon',
    email: 'priya@example.com',
    role: 'Senior Geospatial Analyst',
    organization: 'National Remote Sensing Centre (NRSC)',
    plan: 'Pro Tier',
    avatar: 'P',
    avatar_bg: 'linear-gradient(135deg, #184e42, #73cfb3)',
    api_key: 'sq_live_9a87f12e4b3c7d6e',
    queries_limit: 500,
    queries_used: 42
  };
  updateUIUser(currentUser);
}

function updateUIUser(user) {
  if (!user) {
    document.querySelectorAll('.user-name-target').forEach(el => el.textContent = 'Guest Observer');
    document.querySelectorAll('.user-email-target').forEach(el => el.textContent = 'Not signed in');
    document.querySelectorAll('.user-plan-target').forEach(el => el.textContent = 'Free Community');
    document.querySelectorAll('.user-plan-badge-target').forEach(el => el.textContent = 'Guest Tier');
    document.querySelectorAll('.user-avatar-target').forEach(el => {
      el.textContent = 'G';
      el.style.background = 'linear-gradient(135deg, #5c6b73, #9db4c0)';
    });
    const navText = document.querySelector('.nav-auth-text');
    if (navText) navText.textContent = 'Sign In';
    return;
  }

  const initial = user.avatar || (user.name ? user.name.charAt(0).toUpperCase() : 'P');
  const bg = user.avatar_bg || 'linear-gradient(135deg, #184e42, #73cfb3)';

  document.querySelectorAll('.user-name-target').forEach(el => el.textContent = user.name);
  document.querySelectorAll('.user-avatar-target').forEach(el => {
    el.textContent = initial;
    el.style.background = bg;
  });
  document.querySelectorAll('.user-email-target').forEach(el => el.textContent = user.email);
  document.querySelectorAll('.user-plan-target').forEach(el => el.textContent = `${user.plan} · Settings`);
  document.querySelectorAll('.user-plan-badge-target').forEach(el => el.textContent = user.plan);

  const navText = document.querySelector('.nav-auth-text');
  if (navText) navText.textContent = user.name.split(' ')[0];

  // Update Account Modal Form Fields
  const nameInput = document.getElementById('profile-name-input');
  const emailInput = document.getElementById('profile-email-input');
  const roleInput = document.getElementById('profile-role-input');
  const orgInput = document.getElementById('profile-org-input');
  const keyDisplay = document.getElementById('api-key-display');
  const keySnippet = document.getElementById('api-key-snippet-token');
  const quotaText = document.getElementById('quota-used-text');
  const quotaBar = document.getElementById('quota-bar-fill');

  if (nameInput) nameInput.value = user.name || '';
  if (emailInput) emailInput.value = user.email || '';
  if (roleInput) roleInput.value = user.role || '';
  if (orgInput) orgInput.value = user.organization || '';
  if (keyDisplay) keyDisplay.value = user.api_key || 'sq_live_demo';
  if (keySnippet) keySnippet.textContent = user.api_key || 'sq_live_demo';

  const used = user.queries_used || 0;
  const limit = user.queries_limit || 100;
  if (quotaText) quotaText.textContent = `${used} / ${limit} queries used`;
  if (quotaBar) {
    const pct = Math.min(100, Math.round((used / limit) * 100));
    quotaBar.style.width = `${pct}%`;
  }

  document.querySelectorAll('.persona-card').forEach(card => {
    card.classList.toggle('active-persona', card.dataset.persona === user.id);
  });
}

// Tab switcher for Auth Modal
function switchAuthTab(tabId) {
  document.querySelectorAll('.auth-tab-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tab === tabId);
  });
  document.querySelectorAll('.auth-tab-content').forEach(content => {
    content.style.display = content.id === `auth-tab-${tabId}` ? 'block' : 'none';
  });
  const feedback = document.getElementById('auth-feedback');
  if (feedback) feedback.style.display = 'none';
}

// Tab switcher for Account Modal
function switchAccountTab(tabId) {
  document.querySelectorAll('.account-tab-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tab === tabId);
  });
  document.querySelectorAll('.account-tab-content').forEach(content => {
    content.style.display = content.id === `account-tab-${tabId}` ? 'block' : 'none';
  });
}

// 1-Click Login via Demo Persona
async function loginWithPersona(personaId) {
  const authModal = document.getElementById('auth-modal');
  try {
    const res = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ persona_id: personaId })
    });
    const data = await res.json();
    if (data.success && data.user) {
      authToken = data.token;
      localStorage.setItem('satquery_token', authToken);
      currentUser = data.user;
      updateUIUser(currentUser);
      if (authModal) toggleOverlay(authModal, false);
      notify(`Logged in as ${currentUser.name} (${currentUser.plan})`);
    } else {
      notify(data.error || 'Login failed');
    }
  } catch (err) {
    console.error('Persona login error:', err);
    notify('Failed to connect to authentication service');
  }
}

// ---------------------------------------------------------------------------
// OVERLAY / MODAL MANAGEMENT
// ---------------------------------------------------------------------------
const settingsDrawer = document.getElementById('settings-drawer');
const settingsBackdrop = document.getElementById('settings-backdrop');
const searchSheet = document.getElementById('search-sheet');
const authModal = document.getElementById('auth-modal');
const accountModal = document.getElementById('account-modal');

function toggleOverlay(element, visible) {
  if (!element) return;
  if (typeof gsap !== 'undefined') {
    gsap.killTweensOf(element);
    gsap.set(element, { clearProps: 'transform,opacity,scale' });
  }
  element.style.transform = '';
  element.style.opacity = '';

  if (visible) {
    element.classList.add('open');
    element.setAttribute('aria-hidden', 'false');
    if (element === settingsDrawer && settingsBackdrop) settingsBackdrop.classList.add('open');

    // Animate modal panels
    if (typeof gsap !== 'undefined') {
      if (element === authModal || element === accountModal) {
        const panel = element.querySelector('.pipeline-panel');
        if (panel) {
          gsap.fromTo(panel, { scale: 0.88, opacity: 0, y: 25, rotationX: 6 }, { scale: 1, opacity: 1, y: 0, rotationX: 0, duration: 0.45, ease: 'back.out(1.3)' });
        }
        const flowItems = element.querySelectorAll('.flow article');
        if (flowItems.length > 0) {
          gsap.fromTo(flowItems, { y: 20, opacity: 0 }, { y: 0, opacity: 1, duration: 0.4, stagger: 0.08, ease: 'power2.out', delay: 0.1 });
        }
      } else if (element === settingsDrawer) {
        gsap.fromTo('.settings-drawer', { x: '100%' }, { x: '0%', duration: 0.45, ease: 'power3.out' });
      } else if (element === searchSheet) {
        gsap.fromTo('.search-card', { scale: 0.88, opacity: 0, y: 25 }, { scale: 1, opacity: 1, y: 0, duration: 0.45, ease: 'power3.out' });
      }
    }
  } else {
    element.classList.remove('open');
    element.setAttribute('aria-hidden', 'true');
    if (element === settingsDrawer && settingsBackdrop) settingsBackdrop.classList.remove('open');
  }
}

// ---------------------------------------------------------------------------
// Workspace Scroll Isolation
// ---------------------------------------------------------------------------
function initWorkspaceScrollIsolation() {
  const workspace = document.getElementById('workspace');
  if (!workspace) return;

  workspace.addEventListener('wheel', (e) => {
    e.stopPropagation();
  }, { passive: true });

  workspace.addEventListener('touchmove', (e) => {
    e.stopPropagation();
  }, { passive: true });
}

// ---------------------------------------------------------------------------
// Navigation Handlers (Home / Demo buttons)
// ---------------------------------------------------------------------------
function initNavigationHandlers() {
  const goHomeButtons = document.querySelectorAll('#go-home, a[href="#landing"], .brand');
  goHomeButtons.forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();

      isHomeNavigating = true;
      if (globalStopWaterfallLoop) globalStopWaterfallLoop();
      if (globalRenderFrameZero) globalRenderFrameZero();
      if (toast) toast.classList.remove('show');

      window.scrollTo({ top: 0, behavior: 'instant' });

      if (cinematicTrigger) {
        if (cinematicTrigger.scroll) cinematicTrigger.scroll(cinematicTrigger.start);
        if (cinematicTrigger.animation) cinematicTrigger.animation.progress(0);
      }

      if (typeof gsap !== 'undefined') {
        gsap.set('.workspace', { autoAlpha: 0 });
        gsap.set('.navigation.landing-nav', { autoAlpha: 1 });
        gsap.set('.opening-panel', { autoAlpha: 1, yPercent: 0, filter: 'blur(0px)' });
      }

      deactivateWorkspace();

      setTimeout(() => {
        isHomeNavigating = false;
        if (globalRenderFrameZero) globalRenderFrameZero();
        if (typeof ScrollTrigger !== 'undefined') {
          ScrollTrigger.refresh();
        }
      }, 400);
    });
  });

  const demoButtons = document.querySelectorAll('.start-query, .nav-demo');
  demoButtons.forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();

      isDemoNavigating = true;

      if (typeof gsap !== 'undefined') {
        gsap.set('.opening-panel, .panel-2, .final-panel, .afterglow, .navigation.landing-nav', { autoAlpha: 0 });
        gsap.set('.workspace', { autoAlpha: 1, y: 0, scale: 1 });
      }

      const endScroll = (cinematicTrigger && cinematicTrigger.end) ? cinematicTrigger.end : 2400;
      window.scrollTo({ top: endScroll, behavior: 'instant' });

      if (cinematicTrigger) {
        if (cinematicTrigger.scroll) cinematicTrigger.scroll(cinematicTrigger.end);
        if (cinematicTrigger.animation) cinematicTrigger.animation.progress(1);
      }

      if (globalStartWaterfallLoop) globalStartWaterfallLoop();
      if (typeof playGamingTypewriter === 'function' && !hasTypedWorkspace) {
        hasTypedWorkspace = true;
        playGamingTypewriter();
      }

      activateWorkspace();

      setTimeout(() => {
        isDemoNavigating = false;
        if (typeof ScrollTrigger !== 'undefined') {
          ScrollTrigger.refresh();
        }
      }, 300);
    });
  });
}

// ---------------------------------------------------------------------------
// EVENT BINDINGS (Wired on DOMContentLoaded below)
// ---------------------------------------------------------------------------
function initEventBindings() {
  // Persona Cards
  document.querySelectorAll('.persona-card').forEach(card => {
    card.addEventListener('click', () => {
      const pid = card.dataset.persona;
      loginWithPersona(pid);
    });
  });

  // Auth Form: Sign In
  const signinForm = document.getElementById('signin-form');
  if (signinForm) {
    signinForm.addEventListener('submit', async e => {
      e.preventDefault();
      const email = document.getElementById('signin-email').value.trim();
      const password = document.getElementById('signin-password').value.trim();

      try {
        const res = await fetch('/api/auth/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ email, password })
        });
        const data = await res.json();
        if (data.success && data.user) {
          authToken = data.token;
          localStorage.setItem('satquery_token', authToken);
          currentUser = data.user;
          updateUIUser(currentUser);
          if (authModal) toggleOverlay(authModal, false);
          notify(data.message || `Welcome, ${currentUser.name}`);
        } else {
          const feedback = document.getElementById('auth-feedback');
          if (feedback) {
            feedback.textContent = data.error || 'Invalid credentials';
            feedback.className = 'auth-feedback error';
            feedback.style.display = 'block';
          }
        }
      } catch (err) {
        notify('Connection error during sign in');
      }
    });
  }

  // Auth Form: Create Account
  const signupForm = document.getElementById('signup-form');
  if (signupForm) {
    signupForm.addEventListener('submit', async e => {
      e.preventDefault();
      const name = document.getElementById('signup-name').value.trim();
      const email = document.getElementById('signup-email').value.trim();
      const org = document.getElementById('signup-org').value.trim();

      try {
        const res = await fetch('/api/auth/signup', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name, email, organization: org })
        });
        const data = await res.json();
        if (data.success && data.user) {
          authToken = data.token;
          localStorage.setItem('satquery_token', authToken);
          currentUser = data.user;
          updateUIUser(currentUser);
          if (authModal) toggleOverlay(authModal, false);
          notify(data.message || `Account created for ${currentUser.name}`);
        } else {
          const feedback = document.getElementById('auth-feedback');
          if (feedback) {
            feedback.textContent = data.error || 'Failed to create account';
            feedback.className = 'auth-feedback error';
            feedback.style.display = 'block';
          }
        }
      } catch (err) {
        notify('Connection error during registration');
      }
    });
  }

  // Profile Edit Form Submit
  const profileEditForm = document.getElementById('profile-edit-form');
  if (profileEditForm) {
    profileEditForm.addEventListener('submit', async e => {
      e.preventDefault();
      const name = document.getElementById('profile-name-input').value.trim();
      const email = document.getElementById('profile-email-input').value.trim();
      const role = document.getElementById('profile-role-input').value.trim();
      const org = document.getElementById('profile-org-input').value.trim();

      try {
        const res = await fetch('/api/auth/update-profile', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${authToken}`
          },
          body: JSON.stringify({ name, email, role, organization: org })
        });
        const data = await res.json();
        if (data.success && data.user) {
          currentUser = data.user;
          updateUIUser(currentUser);
          if (accountModal) toggleOverlay(accountModal, false);
          notify('Profile updated successfully');
        } else {
          notify(data.error || 'Failed to update profile');
        }
      } catch (err) {
        notify('Connection error updating profile');
      }
    });
  }

  // Copy API Key
  const copyKeyBtn = document.getElementById('copy-api-key-btn');
  if (copyKeyBtn) {
    copyKeyBtn.addEventListener('click', () => {
      const keyInput = document.getElementById('api-key-display');
      if (keyInput) {
        navigator.clipboard.writeText(keyInput.value).then(() => {
          notify('API Key copied to clipboard');
        }).catch(() => {
          keyInput.select();
          document.execCommand('copy');
          notify('API Key copied to clipboard');
        });
      }
    });
  }

  // Regenerate API Key
  const regenKeyBtn = document.getElementById('regen-api-key-btn');
  if (regenKeyBtn) {
    regenKeyBtn.addEventListener('click', async () => {
      try {
        const res = await fetch('/api/auth/regenerate-key', {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${authToken}`
          }
        });
        const data = await res.json();
        if (data.success && data.api_key) {
          if (currentUser) currentUser.api_key = data.api_key;
          const keyDisplay = document.getElementById('api-key-display');
          const keySnippet = document.getElementById('api-key-snippet-token');
          if (keyDisplay) keyDisplay.value = data.api_key;
          if (keySnippet) keySnippet.textContent = data.api_key;
          notify('New SatQuery API Token generated');
        }
      } catch (err) {
        notify('Failed to regenerate API token');
      }
    });
  }

  // Real Logout Handler
  const logoutBtn = document.getElementById('logout');
  if (logoutBtn) {
    logoutBtn.addEventListener('click', async () => {
      try {
        await fetch('/api/auth/logout', {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${authToken}`
          }
        });
      } catch (e) {
        console.warn('Logout endpoint error:', e);
      }

      localStorage.removeItem('satquery_token');
      authToken = null;
      currentUser = null;
      updateUIUser(null);
      toggleOverlay(settingsDrawer, false);
      notify('You have been logged out.');
      show('landing');
    });
  }

  // Modal tab listeners
  document.querySelectorAll('.auth-tab-btn').forEach(btn => {
    btn.addEventListener('click', () => switchAuthTab(btn.dataset.tab));
  });
  document.querySelectorAll('.account-tab-btn').forEach(btn => {
    btn.addEventListener('click', () => switchAccountTab(btn.dataset.tab));
  });

  // Trigger Buttons
  const openAuthBtn = document.getElementById('open-auth-btn');
  if (openAuthBtn) {
    openAuthBtn.addEventListener('click', () => {
      if (currentUser) {
        switchAccountTab('profile');
        if (accountModal) toggleOverlay(accountModal, true);
      } else {
        switchAuthTab('signin');
        if (authModal) toggleOverlay(authModal, true);
      }
    });
  }

  const headerAvatar = document.getElementById('header-avatar');
  if (headerAvatar) {
    headerAvatar.addEventListener('click', () => {
      switchAccountTab('profile');
      if (accountModal) toggleOverlay(accountModal, true);
    });
  }

  const drawerOpenAccount = document.getElementById('drawer-open-account');
  if (drawerOpenAccount) {
    drawerOpenAccount.addEventListener('click', () => {
      toggleOverlay(settingsDrawer, false);
      switchAccountTab('profile');
      if (accountModal) toggleOverlay(accountModal, true);
    });
  }

  const drawerOpenPreferences = document.getElementById('drawer-open-preferences');
  if (drawerOpenPreferences) {
    drawerOpenPreferences.addEventListener('click', () => {
      toggleOverlay(settingsDrawer, false);
      switchAccountTab('quota');
      if (accountModal) toggleOverlay(accountModal, true);
    });
  }

  const drawerSwitchPersona = document.getElementById('drawer-switch-persona');
  if (drawerSwitchPersona) {
    drawerSwitchPersona.addEventListener('click', () => {
      toggleOverlay(settingsDrawer, false);
      switchAuthTab('demo');
      if (authModal) toggleOverlay(authModal, true);
    });
  }

  const switchToDemoLink = document.getElementById('switch-to-demo-link');
  if (switchToDemoLink) {
    switchToDemoLink.addEventListener('click', () => switchAuthTab('demo'));
  }

  // Close Buttons for Modals
  const closeAuth = document.getElementById('close-auth');
  if (closeAuth) closeAuth.addEventListener('click', () => toggleOverlay(authModal, false));
  if (authModal) authModal.addEventListener('click', e => { if (e.target === authModal) toggleOverlay(authModal, false); });

  const closeAccount = document.getElementById('close-account');
  if (closeAccount) closeAccount.addEventListener('click', () => toggleOverlay(accountModal, false));
  if (accountModal) accountModal.addEventListener('click', e => { if (e.target === accountModal) toggleOverlay(accountModal, false); });

  // Core Chat & Toolbar
  document.getElementById('submit-query').addEventListener('click', submitChat);
  question.addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); submitChat(); } });
  document.getElementById('locate').addEventListener('click', () => { locationInput.value = 'Chennai, Tamil Nadu, India'; notify('Location updated to Chennai.'); });

  // Artifact Side Panel Controls
  const toggleArtifactBtn = document.getElementById('toggle-artifact-pane-btn');
  if (toggleArtifactBtn) {
    toggleArtifactBtn.addEventListener('click', () => {
      const pane = document.getElementById('artifact-pane');
      if (pane && pane.classList.contains('open')) {
        closeArtifactPane();
      } else {
        openArtifactPane();
      }
    });
  }
  const artifactCloseBtn = document.getElementById('artifact-close-btn');
  if (artifactCloseBtn) artifactCloseBtn.addEventListener('click', closeArtifactPane);

  const artifactFullscreenBtn = document.getElementById('artifact-fullscreen-btn');
  if (artifactFullscreenBtn) artifactFullscreenBtn.addEventListener('click', toggleArtifactFullscreen);

  document.getElementById('open-settings').addEventListener('click', (e) => { e.stopPropagation(); toggleOverlay(settingsDrawer, true); });
  document.getElementById('close-settings').addEventListener('click', (e) => { e.preventDefault(); e.stopPropagation(); toggleOverlay(settingsDrawer, false); });
  if (settingsBackdrop) settingsBackdrop.addEventListener('click', () => toggleOverlay(settingsDrawer, false));
  document.getElementById('open-search').addEventListener('click', () => toggleOverlay(searchSheet, true));
  document.getElementById('close-search').addEventListener('click', (e) => { e.preventDefault(); toggleOverlay(searchSheet, false); });
  document.getElementById('new-chat').addEventListener('click', () => {
    thread.innerHTML = '';
    welcome.hidden = false;
    currentPresetId = null;
    document.querySelectorAll('.preset-card').forEach(c => c.classList.remove('active'));
    attachedFiles = [];
    document.getElementById('attachment-status').textContent = '';
    closeArtifactPane();
    const tBtn = document.getElementById('toggle-artifact-pane-btn');
    if (tBtn) tBtn.style.display = 'none';
    question.focus();
  });
  document.getElementById('open-images').addEventListener('click', () => notify('Your 12 stored satellite images are ready to browse.'));

  // Architecture & Technical Verification Modal Wiring
  const archModal = document.getElementById('architecture-modal');
  const openArchBtn = document.getElementById('open-architecture-btn');
  const closeArchBtn = document.getElementById('close-architecture');
  const closeArchBtn2 = document.getElementById('close-architecture-btn');

  const closeArch = () => {
    if (archModal) {
      archModal.style.display = 'none';
      archModal.classList.remove('open');
    }
  };

  if (openArchBtn && archModal) {
    openArchBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      archModal.style.display = 'flex';
      archModal.classList.add('open');
    });
  }
  if (closeArchBtn) closeArchBtn.addEventListener('click', closeArch);
  if (closeArchBtn2) closeArchBtn2.addEventListener('click', closeArch);
  if (archModal) {
    archModal.addEventListener('click', (e) => {
      if (e.target === archModal) closeArch();
    });
  }

  // Image file input handling
  const imageInput = document.getElementById('image-input');
  if (imageInput) {
    imageInput.addEventListener('change', e => {
      attachedFiles = Array.from(e.target.files);
      if (attachedFiles.length > 0) {
        currentPresetId = null;
        const names = attachedFiles.map(f => f.name).join(', ');
        document.getElementById('attachment-status').textContent = `✓ ${attachedFiles.length} file(s) attached: ${names}`;
        notify(`${attachedFiles.length} image(s) attached successfully.`);
      }
    });
  }

  // Suggestion chips
  document.querySelectorAll('.suggestions button').forEach(chip => chip.addEventListener('click', () => { question.value = chip.textContent; question.focus(); }));

  // Settings drawer close on outside click
  document.addEventListener('click', (e) => {
    if (settingsDrawer && settingsDrawer.classList.contains('open')) {
      if (!settingsDrawer.contains(e.target) && !e.target.closest('#open-settings')) {
        toggleOverlay(settingsDrawer, false);
      }
    }
  });

  // Escape key handler
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      const artPane = document.getElementById('artifact-pane');
      if (artPane && artPane.classList.contains('fullscreen')) {
        artPane.classList.remove('fullscreen');
      } else if (artPane && artPane.classList.contains('open')) {
        closeArtifactPane();
      }
      if (settingsDrawer && settingsDrawer.classList.contains('open')) toggleOverlay(settingsDrawer, false);
      if (searchSheet && searchSheet.classList.contains('open')) toggleOverlay(searchSheet, false);
      if (authModal && authModal.classList.contains('open')) toggleOverlay(authModal, false);
      if (accountModal && accountModal.classList.contains('open')) toggleOverlay(accountModal, false);
    }
  });
}

// ---------------------------------------------------------------------------
// Intro Splash Video Screen (Pre-Site Launch from ui/preview)
// ---------------------------------------------------------------------------
function initIntroVideo() {
  const overlay = document.getElementById('intro-video-overlay');
  const video = document.getElementById('intro-video');
  const skipBtn = document.getElementById('intro-skip-btn');
  const soundBtn = document.getElementById('intro-sound-btn');
  if (!overlay || !video) return;

  let hasDismissed = false;

  const dismissIntro = () => {
    if (hasDismissed) return;
    hasDismissed = true;
    try {
      video.pause();
    } catch (e) {}

    if (typeof gsap !== 'undefined') {
      gsap.to(overlay, {
        opacity: 0,
        duration: 0.65,
        ease: 'power2.inOut',
        onComplete: () => {
          overlay.classList.add('hidden');
          overlay.style.display = 'none';
        }
      });
    } else {
      overlay.classList.add('hidden');
      setTimeout(() => {
        overlay.style.display = 'none';
      }, 650);
    }
  };

  if (soundBtn) {
    soundBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      video.muted = !video.muted;
      if (video.muted) {
        soundBtn.innerHTML = '🔇 <span>Muted</span>';
      } else {
        soundBtn.innerHTML = '🔊 <span>Sound On</span>';
      }
    });
  }

  if (skipBtn) {
    skipBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      dismissIntro();
    });
  }

  video.addEventListener('ended', dismissIntro);
  video.addEventListener('error', () => {
    console.warn('Intro video error, dismissing overlay.');
    dismissIntro();
  });

  video.play().catch(err => {
    console.log('Intro video autoplay blocked:', err);
    overlay.addEventListener('click', () => {
      video.play().catch(() => dismissIntro());
    }, { once: true });
  });
}

// ---------------------------------------------------------------------------
// DOMContentLoaded — Initialize Everything
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
  initIntroVideo();
  initScrollTrigger();
  init3DMotion();
  initBeforeAfterSliders();
  initMultiBandSelectors();
  initWorkspaceScrollIsolation();
  initNavigationHandlers();
  initPresets();
  initAuth();
  initEventBindings();
});
