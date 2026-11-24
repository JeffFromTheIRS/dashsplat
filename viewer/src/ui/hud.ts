// Overlay modelled element by element on Tesla's Reality View screen (reference/ref_15.5.jpg and friends):
// bare white text on the scene top left, status top right, dark translucent cards along the bottom. The drive
// timeline takes the place of the media card, the trip card is filled from the recording, and the map mode,
// scene, paint and view pickers live in a menu off the trip card's "...".

import type { Drive, DriveState } from "../scene/drive";
import type { Turn } from "../render/path";
import type { ViewMode } from "../render/splatShader";
import type { CamMode } from "../camera/controller";
import type { Overlay } from "../render/generative";
import type { PhotoSwitches } from "../render/photo";
import { PAINTS } from "../render/car";
import { icons } from "./icons";

export type SceneEntry = { name: string; url: string; hasMap: boolean; tiles: number; splats: number };
export type StreetEntry = { kind: "turn" | "end" | "start"; s: number; lat: number; lon: number; road: string; city: string };

export type HudEvents = {
  playPause(): void;
  seek(t: number): void;
  skip(dt: number): void;
  rate(): void;
  mode(m: ViewMode): void;
  scene(url: string): void;
  map(): void;
  night(): void;
  camera(): void;
  cameraStep(d: number): void;
  view(m: CamMode): void;
  paint(id: string): void;
  colour(c: "natural" | "camera"): void;
  overlay(o: Overlay): void;
  predicted(on: boolean): void;
  photoLayer(k: keyof PhotoSwitches): void;
  ground(on: boolean): void;
  vehicles(clean: boolean): void;
  endTrip(): void;
};

const MPS_TO_MPH = 2.23694;
const M_PER_MI = 1609.344;
const BATTERY_START = 78; // % at the start of the recording (not in the telemetry; shown as Tesla would)
const BATTERY_PER_KM = 0.21;

function el<K extends keyof HTMLElementTagNameMap>(tag: K, cls?: string, html?: string) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html !== undefined) e.innerHTML = html;
  return e;
}
export function fmtClock(d: Date) {
  let h = d.getHours();
  const ap = h >= 12 ? "pm" : "am";
  h = h % 12 || 12;
  return `${h}:${String(d.getMinutes()).padStart(2, "0")} ${ap}`;
}
const fmtT = (t: number) => {
  const s = Math.max(0, Math.floor(t));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
};
const fmtDist = (m: number) => (m < 0.1 * M_PER_MI ? `${Math.max(50, Math.round((m * 3.28084) / 50) * 50)} ft` : `${(m / M_PER_MI).toFixed(1)} mi`);
const ABBR: [RegExp, string][] = [[/\bParkway\b/, "Pkwy"], [/\bDrive\b/, "Dr"], [/\bAvenue\b/, "Ave"], [/\bStreet\b/, "St"], [/\bBoulevard\b/, "Blvd"], [/\bRoad\b/, "Rd"], [/\bLane\b/, "Ln"], [/\bCourt\b/, "Ct"], [/\bHighway\b/, "Hwy"], [/\bNorth\b/, "N"], [/\bSouth\b/, "S"]];
const abbr = (s: string) => ABBR.reduce((a, [r, v]) => a.replace(r, v), s);
const battery = (sMeters: number) => Math.round(BATTERY_START - (BATTERY_PER_KM * sMeters) / 1000);
const battGlyph = (pct: number) => `<span class="batt"><span style="width:${Math.max(6, Math.min(100, pct))}%"></span></span>`;

const MODE_LABEL: Record<ViewMode, string> = { live: "Live", stitched: "Stitched", memory: "Memory", generative: "Generative" };

export class Hud {
  readonly root = el("div", "hud");
  private gears: HTMLElement[] = [];
  private battPct = el("span", "batt-pct");
  private battIcon = el("span", "batt-wrap");
  private speed = el("div", "speed-num");
  private sdTitle = el("div", "sd-title", "Self-Driving");
  private sdMode = el("div", "sd-mode");
  private sdSub = el("div", "sd-sub");
  private streakNum = el("div", "streak-num");
  private limit = el("div", "limit-num");
  private navIcon = el("div", "nav-icon");
  private navDist = el("div", "nav-dist");
  private navText = el("div", "nav-text");
  private clock = el("div", "clock");
  private status = el("div", "status");
  private camLabel = el("div", "cam-label");
  private mapBtn = el("button", "btn map-btn") as HTMLButtonElement;
  private nightBtn = el("button", "btn sq-btn") as HTMLButtonElement;
  private art = el("canvas", "art") as HTMLCanvasElement;
  private mTitle = el("div", "m-title");
  private mSub = el("div", "m-sub");
  private mTime = el("div", "m-time");
  private track = el("canvas", "track") as HTMLCanvasElement;
  private playBtn = el("button", "ctl") as HTMLButtonElement;
  private rateBtn = el("button", "ctl rate") as HTMLButtonElement;
  private memBtn = el("button", "ctl") as HTMLButtonElement;
  private camBtn = el("button", "ctl") as HTMLButtonElement;
  private tripArrive = el("div", "t-arrive");
  private tripRemain = el("div", "t-remain");
  private tripDest = el("div", "t-dest");
  private tripBatt = el("div", "t-batt");
  private menu = el("div", "menu card");
  private legend = el("div", "legend card");
  private stats = el("div", "stats");
  private loading = el("div", "loading");
  private drive?: Drive;
  private turns: Turn[] = [];
  private limits?: Uint8Array;
  private streets: StreetEntry[] = [];
  private start = new Date(2026, 10, 21, 7, 15, 40);
  private engageS: Float64Array = new Float64Array(0);
  private dragging = false;
  private last = { speed: -1, gear: -1, ap: -2, clock: "", tNow: "", nav: "", limit: -1, streak: "", trip: "", batt: -1, art: -1 };
  private T = 0;
  private sNow = 0;
  private mode: ViewMode = "live";
  private sceneName = "";
  private city = "";
  private scenes: SceneEntry[] = [];
  private sceneUrl = "";
  private camMode: CamMode = "chase";

  constructor(private readonly ev: HudEvents) {
    const r = this.root;
    r.append(el("div", "scrim"));

    // top left: gear and battery, speed, self-driving, streak, speed limit, lights
    const tl = el("div", "tl");
    const gl = el("div", "gearline");
    const gears = el("div", "gears");
    for (const g of ["P", "R", "N", "D"]) {
      const s = el("span", "", g);
      this.gears.push(s);
      gears.append(s);
    }
    gl.append(gears, this.battIcon, this.battPct);
    this.sdMode.innerHTML = `<span>Standard</span>${icons.chevronDown}`;
    const streak = el("div", "streak");
    streak.append(this.streakNum, el("div", "streak-label", "Streak"));
    const lim = el("div", "limit");
    lim.append(el("div", "limit-label", "SPEED<br>LIMIT"), this.limit);
    const lights = el("div", "lights", icons.lowBeam + icons.parkLights);
    tl.append(gl, this.speed, el("div", "speed-unit", "MPH"), this.sdTitle, this.sdMode, this.sdSub, el("div", "rule"), streak, lim, lights);

    const nav = el("div", "nav");
    const navTxt = el("div", "nav-txt");
    navTxt.append(this.navDist, this.navText);
    nav.append(this.navIcon, navTxt);

    // top centre: lock, then our status / camera label
    const tc = el("div", "tc");
    tc.append(el("div", "lock", icons.lock), this.status, this.camLabel);
    this.camLabel.onclick = (e) => this.ev.cameraStep((e as MouseEvent).offsetX < this.camLabel.clientWidth / 2 ? -1 : 1);

    // top right: clock, weather, dot, airbag badge
    const tr = el("div", "tr");
    tr.append(this.clock, el("div", "weather", `${icons.cloud}<span>58°F</span>`), el("div", "dot"), el("div", "airbag", `${icons.airbag}<div><div>PASSENGER</div><div>AIRBAG <b>ON</b></div></div>`));

    // bottom left
    const bl = el("div", "bl");
    this.mapBtn.innerHTML = `${icons.map}<span>Map</span>`;
    this.mapBtn.onclick = () => this.ev.map();
    this.nightBtn.innerHTML = icons.moon;
    this.nightBtn.onclick = () => this.ev.night();
    const moreBtn = el("button", "btn sq-btn more-btn", icons.dots) as HTMLButtonElement;
    moreBtn.onclick = (e) => {
      e.stopPropagation();
      this.menu.classList.toggle("open");
    };
    bl.append(this.mapBtn, this.nightBtn, moreBtn);

    // media card: the drive timeline
    const media = el("div", "card media");
    const info = el("div", "m-info");
    info.append(this.mTitle, this.mSub);
    const controls = el("div", "m-controls");
    const back = el("button", "ctl", icons.prev) as HTMLButtonElement;
    back.onclick = () => this.ev.skip(-10);
    back.title = "Back 10 s (←)";
    this.playBtn.innerHTML = icons.pause;
    this.playBtn.onclick = () => this.ev.playPause();
    this.playBtn.title = "Play / pause (space)";
    const fwd = el("button", "ctl", icons.next) as HTMLButtonElement;
    fwd.onclick = () => this.ev.skip(10);
    fwd.title = "Forward 10 s (→)";
    this.rateBtn.onclick = () => this.ev.rate();
    this.rateBtn.title = "Playback speed (↑ ↓)";
    this.memBtn.innerHTML = icons.memory;
    this.memBtn.onclick = () => this.ev.mode(this.mode === "memory" ? "live" : "memory");
    this.memBtn.title = "Memory: seen now vs remembered (M)";
    this.camBtn.innerHTML = icons.camera;
    this.camBtn.onclick = () => this.ev.camera();
    this.camBtn.title = "Car cameras (3)";
    controls.append(back, this.playBtn, fwd, this.rateBtn, this.memBtn, this.camBtn);
    media.append(this.art, info, this.mTime, this.track, controls);

    // trip card
    const trip = el("div", "card trip");
    const t1 = el("div", "t-row");
    t1.append(this.tripArrive, this.tripRemain);
    const t2 = el("div", "t-row");
    t2.append(this.tripDest, this.tripBatt);
    const t3 = el("div", "t-actions");
    const pin = el("button", "ctl", icons.pinPlus);
    pin.title = "Drop a pin";
    const end = el("button", "end-trip", "End Trip") as HTMLButtonElement;
    end.onclick = () => this.ev.endTrip();
    const more = el("button", "ctl", icons.dots) as HTMLButtonElement;
    more.onclick = (e) => {
      e.stopPropagation();
      this.menu.classList.toggle("open");
    };
    more.title = "Map mode, scene, paint, view";
    t3.append(pin, end, more);
    trip.append(t1, t2, t3);

    this.menu.addEventListener("pointerdown", (e) => e.stopPropagation());
    window.addEventListener("pointerdown", () => this.menu.classList.remove("open"));

    this.legend.innerHTML = `<div class="lg-title">${icons.memory}<span>Memory</span></div>
      <div class="lg-ramp"><span style="background:#8c9a8a"></span><span style="background:#8a9cac"></span><span style="background:#73869c"></span><span style="background:#5c6984"></span><span class="lg-pred"></span></div>
      <div class="lg-labels"><span>Seen now</span><span>10 s</span><span>1 min+</span><span class="lg-pred-l">Predicted</span></div>
      <div class="lg-note">Outlines show what the cameras see right now. Predicted: never seen, inferred from continuity or motion</div>`;

    r.append(tl, nav, tc, tr, bl, media, trip, this.menu, this.legend, this.stats, this.loading);
    this.setupTrack();
    this.stats.hidden = true;
    this.legend.hidden = true;
    this.camLabel.hidden = true;
    this.renderMenu();
  }

  setScenes(list: SceneEntry[], current: string) {
    this.scenes = list;
    this.sceneUrl = current;
    this.sceneName = current.replace(/^\/scenes\/|\/$/g, "");
    this.renderTitles();
    this.renderMenu();
  }

  setStreets(entries: StreetEntry[]) {
    this.streets = entries;
  }

  private renderTitles() {
    const label = this.mode === "live" && /stitch/i.test(this.sceneName) ? "Stitched" : MODE_LABEL[this.mode];
    this.mTitle.textContent = `${this.city || "Drive"} · ${label}`;
    this.mSub.innerHTML = `<span class="m-dot"></span><span>${this.sceneName.toUpperCase()}</span>`;
  }

  private renderMenu() {
    const m = this.menu;
    m.innerHTML = "";
    const section = (title: string) => {
      const s = el("div", "mn-sec");
      s.append(el("div", "mn-title", title));
      m.append(s);
      return s;
    };
    const seg = (parent: HTMLElement, items: [string, string][], on: string, pick: (v: string) => void) => {
      const g = el("div", "mn-seg");
      for (const [v, label] of items) {
        const b = el("button", v === on ? "on" : "", label) as HTMLButtonElement;
        b.onclick = () => pick(v);
        g.append(b);
      }
      parent.append(g);
    };
    const modes: [string, string][] = [["live", "Live"], ["stitched", "Stitched"], ["memory", "Memory"]];
    if (this.generative) modes.push(["generative", "Generative"]);
    seg(section("Map"), modes, this.mode, (v) => this.ev.mode(v as ViewMode));
    if (this.generative && this.mode === "generative") {
      seg(section("Over the generated video"), [["auto", "Auto"], ["car", "Car + path"], ["path", "Path"], ["none", "Nothing"]], this.overlay, (v) => this.ev.overlay(v as Overlay));
    }
    // what was predicted rather than observed: the generated clip's provenance mask, the textured ground's mask
    if (this.predictedAvailable) seg(section("Predicted"), [["off", "Off"], ["on", "Show"]], this.predicted ? "on" : "off", (v) => this.ev.predicted(v === "on"));
    if (this.vehiclesClean !== null) seg(section("Vehicles"), [["clean", "Clean"], ["scanned", "Scanned"]], this.vehiclesClean ? "clean" : "scanned", (v) => this.ev.vehicles(v === "clean"));
    if (this.ground !== null) seg(section("Ground"), [["atlas", "Textured"], ["splats", "Splats"]], this.ground ? "atlas" : "splats", (v) => this.ev.ground(v === "atlas"));
    if (this.photo) {
      const ps = section(`Photo layers · ${this.photo.source}`);
      const g = el("div", "mn-seg");
      for (const [k, label] of [["sky", "Sky"], ["backdrop", "Backdrop"], ["cutouts", "Cut-outs"], ["movers", "Movers"]] as [keyof PhotoSwitches, string][]) {
        const b = el("button", this.photo.on[k] ? "on" : "", label) as HTMLButtonElement;
        b.onclick = () => this.ev.photoLayer(k);
        g.append(b);
      }
      ps.append(g);
    }
    seg(section("View"), [["chase", "Follow"], ["map", "Map"], ["cams", "Cameras"], ["free", "Free"]], this.camMode, (v) => this.ev.view(v as CamMode));
    seg(section("Colour"), [["natural", "Natural"], ["camera", "Camera"]], this.colourMode, (v) => this.ev.colour(v as "natural" | "camera"));
    const ps = section("Paint");
    const sw = el("div", "mn-paint");
    for (const p of PAINTS) {
      const b = el("button", "", "") as HTMLButtonElement;
      b.style.setProperty("--c", p.color);
      b.title = p.name;
      b.dataset.id = p.id;
      if (p.id === this.paintId) b.classList.add("on");
      b.onclick = () => {
        this.ev.paint(p.id);
        sw.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
      };
      sw.append(b);
    }
    ps.append(sw);
    const ss = section("Scene");
    const list = el("div", "mn-list");
    for (const s of this.scenes) {
      const b = el("button", s.url === this.sceneUrl ? "on" : "", `<span>${s.name}</span><span class="mn-meta">${s.hasMap ? (s.splats / 1e6).toFixed(1) + "M" : "no tiles"}</span>`) as HTMLButtonElement;
      b.onclick = () => {
        this.menu.classList.remove("open");
        this.ev.scene(s.url);
      };
      list.append(b);
    }
    ss.append(list);
  }

  private ground: boolean | null = null;
  private vehiclesClean: boolean | null = null;
  /** the scene has vehicles: offer Clean / Scanned (null: none) */
  setVehicles(clean: boolean | null) {
    this.vehiclesClean = clean;
    this.renderMenu();
  }
  /** the scene has a textured ground: offer the switch (null: none) */
  setGround(on: boolean | null) {
    this.ground = on;
    this.renderMenu();
  }
  private photo: { source: string; on: PhotoSwitches } | null = null;
  /** the scene has photographic layers (from itself or borrowed): offer their switches */
  setPhoto(p: { source: string; on: PhotoSwitches } | null) {
    this.photo = p && { source: p.source, on: { ...p.on } };
    this.renderMenu();
  }
  private generative = false;
  /** the scene has generated video: offer the Generative mode */
  setGenerative(on: boolean) {
    this.generative = on;
    this.renderMenu();
  }
  private predicted = false;
  private predictedAvailable = false;
  setPredicted(on: boolean, available?: boolean) {
    this.predicted = on;
    if (available !== undefined) this.predictedAvailable = available;
    this.renderMenu();
  }
  private overlay: Overlay = "auto";
  setOverlay(o: Overlay) {
    this.overlay = o;
    this.renderMenu();
  }

  private colourMode = "natural";
  setColour(c: string) {
    this.colourMode = c;
    this.renderMenu();
  }

  private paintId = "";
  setPaint(id: string) {
    this.paintId = id;
    this.menu.querySelectorAll<HTMLButtonElement>(".mn-paint button").forEach((b) => b.classList.toggle("on", b.dataset.id === id));
  }

  setDrive(drive: Drive, turns: Turn[], limits: Uint8Array) {
    this.drive = drive;
    this.turns = turns;
    this.limits = limits;
    this.engageS = new Float64Array(drive.n);
    let s0 = -1;
    for (let i = 0; i < drive.n; i++) {
      if (drive.ap[i] === 1 && (i === 0 || drive.ap[i - 1] !== 1)) s0 = drive.s[i];
      if (drive.ap[i] !== 1) s0 = -1;
      this.engageS[i] = s0;
    }
    const arrive = new Date(this.start.getTime() + drive.duration * 1000);
    this.tripArrive.textContent = fmtClock(arrive);
    const end = this.street("end", drive.length);
    const st = this.street("start", 0);
    this.city = st?.city || end?.city || "";
    this.tripDest.textContent = end ? abbr(end.road) : "End of recording";
    this.tripBatt.innerHTML = `${battGlyph(battery(drive.length))}<span>${battery(drive.length)}%</span>`;
    this.renderTitles();
    this.last.art = -1;
    this.trackBase = undefined;
  }

  /** street entry for a place on this drive, if the stored position agrees with the drive's own GPS */
  private street(kind: StreetEntry["kind"], s: number) {
    const d = this.drive;
    let best: StreetEntry | undefined;
    for (const e of this.streets) {
      if (e.kind !== kind || Math.abs(e.s - s) > 60) continue;
      const ll = d?.latLonAtS(e.s);
      if (ll && Math.hypot((ll[0] - e.lat) * 111e3, (ll[1] - e.lon) * 96e3) > 120) continue;
      best = e;
    }
    return best;
  }

  private setupTrack() {
    const seek = (e: PointerEvent) => {
      if (!this.drive) return;
      const r = this.track.getBoundingClientRect();
      this.ev.seek(Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)) * this.drive.duration);
    };
    this.track.addEventListener("pointerdown", (e) => {
      this.dragging = true;
      this.track.setPointerCapture(e.pointerId);
      seek(e);
    });
    this.track.addEventListener("pointermove", (e) => this.dragging && seek(e));
    const up = () => (this.dragging = false);
    this.track.addEventListener("pointerup", up);
    this.track.addEventListener("pointercancel", up);
  }
  get scrubbing() {
    return this.dragging;
  }

  setNight(on: boolean) {
    this.nightBtn.classList.toggle("on", on);
  }
  setPlaying(p: boolean) {
    this.playBtn.innerHTML = p ? icons.pause : icons.play;
  }
  setRate(r: number) {
    this.rateBtn.textContent = r === 0.5 ? "½×" : `${r}×`;
  }
  setMode(m: ViewMode) {
    this.mode = m;
    this.memBtn.classList.toggle("on", m === "memory");
    this.legend.hidden = m !== "memory";
    this.renderTitles();
    this.renderMenu();
  }
  setCamMode(m: CamMode, label: string) {
    this.camMode = m;
    this.mapBtn.classList.toggle("on", m === "map");
    this.camBtn.classList.toggle("on", m === "cams");
    this.camLabel.hidden = m !== "cams" && m !== "free";
    this.camLabel.innerHTML = m === "cams" ? `<span class="arrow">‹</span><span>${label}</span><span class="arrow">›</span>` : `<span>${label}</span>`;
    this.renderMenu();
  }
  setStatus(text: string) {
    this.status.textContent = text;
    this.status.classList.toggle("show", !!text);
  }
  setLoading(text: string | null) {
    this.loading.textContent = text ?? "";
    this.loading.classList.toggle("show", !!text);
  }
  toggleStats() {
    this.stats.hidden = !this.stats.hidden;
  }
  get statsVisible() {
    return !this.stats.hidden;
  }
  setStats(text: string) {
    if (!this.stats.hidden) this.stats.textContent = text;
  }

  update(st: DriveState) {
    const d = this.drive;
    if (!d) return;
    this.T = st.t;
    this.sNow = st.s;
    const L = this.last;
    const mph = Math.round(st.speed * MPS_TO_MPH);
    if (mph !== L.speed) this.speed.textContent = String((L.speed = mph));
    // Tesla dashcam gear enum: 0 park, 1 drive, 2 reverse, 3 neutral
    const gi = ({ 0: 0, 1: 3, 2: 1, 3: 2 } as Record<number, number>)[st.gear] ?? 3;
    if (gi !== L.gear) {
      L.gear = gi;
      this.gears.forEach((g, i) => g.classList.toggle("on", i === gi));
    }
    const b = battery(st.s);
    if (b !== L.batt) {
      L.batt = b;
      this.battIcon.innerHTML = battGlyph(b);
      this.battPct.textContent = `${b} %`;
    }
    if (st.ap !== L.ap) {
      L.ap = st.ap;
      const on = st.ap === 1 || st.ap === 2;
      this.sdTitle.classList.toggle("on", on);
      this.sdTitle.textContent = st.ap === 2 ? "Autosteer" : st.ap === 3 ? "Cruise" : "Self-Driving";
      this.sdMode.classList.toggle("dim", !on);
      this.sdSub.textContent = on ? "Street" : "Available";
    }
    const eng = this.engageS[d.index(st.t)];
    const streak = `${eng >= 0 ? ((st.s - eng) / M_PER_MI).toFixed(1) : "0.0"} mi`;
    if (streak !== L.streak) this.streakNum.textContent = L.streak = streak;
    const lim = this.limits ? this.limits[Math.min(this.limits.length - 1, Math.max(0, Math.round(st.s)))] : 25;
    if (lim !== L.limit) this.limit.textContent = String((L.limit = lim));
    // next turn and the street it leads onto
    const next = this.turns.find((t) => t.s > st.s + 4);
    let nav: string;
    if (next) {
      const name = this.street("turn", next.s + 35);
      const txt = name?.road ? abbr(name.road) : next.dir > 0 ? "Turn left" : "Turn right";
      const dist = next.s - st.s < 30 ? "Now" : fmtDist(next.s - st.s);
      nav = `${next.dir > 0 ? "left" : "right"}|${dist}|${txt}`;
    } else {
      const end = this.street("end", d.length);
      nav = `arrive|${fmtDist(Math.max(0, d.length - st.s))}|${end?.road ? abbr(end.road) : "Arrive"}`;
    }
    if (nav !== L.nav) {
      L.nav = nav;
      const [ic, dist, txt] = nav.split("|");
      this.navIcon.innerHTML = icons[ic as "left" | "right" | "arrive"];
      this.navDist.textContent = dist;
      this.navText.textContent = txt;
    }
    const now = new Date(this.start.getTime() + st.t * 1000);
    const clock = fmtClock(now);
    if (clock !== L.clock) this.clock.textContent = L.clock = clock;
    const tn = fmtT(st.t);
    if (tn !== L.tNow) {
      L.tNow = tn;
      this.mTime.textContent = `${tn} / ${fmtT(d.duration)}`;
    }
    const trip = `${Math.max(0, Math.ceil((d.duration - st.t) / 60))} min|${(Math.max(0, d.length - st.s) / M_PER_MI).toFixed(1)} mi`;
    if (trip !== L.trip) {
      L.trip = trip;
      const [a, bb] = trip.split("|");
      this.tripRemain.innerHTML = `<span>${a}</span><span>${bb}</span>`;
    }
    const artKey = Math.round(st.s / 8);
    if (artKey !== L.art) {
      L.art = artKey;
      this.drawArt();
    }
    this.drawTrack();
  }

  /** "album art": the route, driven part in blue, current position */
  private drawArt() {
    const d = this.drive;
    if (!d) return;
    const c = this.art;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.max(1, Math.round(c.clientWidth * dpr)), h = Math.max(1, Math.round(c.clientHeight * dpr));
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    const g = c.getContext("2d")!;
    g.fillStyle = "#1e2a36";
    g.fillRect(0, 0, w, h);
    let mnx = Infinity, mny = Infinity, mxx = -Infinity, mxy = -Infinity;
    for (let i = 0; i < d.n; i += 20) { mnx = Math.min(mnx, d.x[i]); mxx = Math.max(mxx, d.x[i]); mny = Math.min(mny, d.y[i]); mxy = Math.max(mxy, d.y[i]); }
    const pad = 0.16 * w;
    const sc = Math.min((w - 2 * pad) / (mxx - mnx || 1), (h - 2 * pad) / (mxy - mny || 1));
    const ox = (w - (mxx - mnx) * sc) / 2, oy = (h - (mxy - mny) * sc) / 2;
    const P = (i: number): [number, number] => [ox + (d.x[i] - mnx) * sc, h - (oy + (d.y[i] - mny) * sc)];
    const now = d.indexAtS(this.sNow);
    g.lineCap = g.lineJoin = "round";
    g.lineWidth = 0.075 * w;
    g.strokeStyle = "rgba(255,255,255,0.28)";
    g.beginPath();
    for (let i = 0; i < d.n; i += 12) { const p = P(i); i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1]); }
    g.stroke();
    g.strokeStyle = "#3e7bff";
    g.beginPath();
    for (let i = 0; i <= now; i += 12) { const p = P(i); i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1]); }
    const pn = P(now);
    g.lineTo(pn[0], pn[1]);
    g.stroke();
    g.fillStyle = "#fff";
    g.beginPath();
    g.arc(pn[0], pn[1], 0.065 * w, 0, Math.PI * 2);
    g.fill();
  }

  private trackBase?: HTMLCanvasElement;
  /** progress line along the card, like the track progress of the media card: played white, handle */
  private drawTrack() {
    const d = this.drive;
    if (!d) return;
    const c = this.track;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.max(1, Math.round(c.clientWidth * dpr)), h = Math.max(1, Math.round(c.clientHeight * dpr));
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; this.trackBase = undefined; }
    const lw = Math.max(2, Math.round(h * 0.13));
    const y = Math.round(h / 2 - lw / 2);
    if (!this.trackBase) {
      const b = document.createElement("canvas");
      b.width = w;
      b.height = h;
      const g = b.getContext("2d")!;
      g.fillStyle = "rgba(255,255,255,0.22)";
      g.fillRect(0, y, w, lw);
      // self-driving stretches, quietly in Tesla blue
      g.fillStyle = "rgba(62,123,255,0.75)";
      let x0 = -1;
      for (let x = 0; x <= w; x++) {
        const on = x < w && d.ap[d.index((x / w) * d.duration)] === 1;
        if (on && x0 < 0) x0 = x;
        if (!on && x0 >= 0) { g.fillRect(x0, y, x - x0, lw); x0 = -1; }
      }
      this.trackBase = b;
    }
    const g = c.getContext("2d")!;
    g.clearRect(0, 0, w, h);
    g.drawImage(this.trackBase, 0, 0);
    const x = (this.T / d.duration) * w;
    g.fillStyle = "#fff";
    g.fillRect(0, y, x, lw);
    g.beginPath();
    g.arc(Math.min(w - h * 0.32, Math.max(h * 0.32, x)), h / 2, h * 0.3, 0, Math.PI * 2);
    g.fill();
  }
}
