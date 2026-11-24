// Outlines of the camera fields of view (six on HW4, four on HW3), drawn from the car in Memory mode: they explain at a glance why
// a place is remembered rather than currently seen. Border rays use each camera's fisheye model (equidistant,
// distortion ignored) so the 174-degree rear camera draws as the cone it is.

import * as THREE from "three";
import type { RigJson } from "../scene/rig";

export class Frustums {
  readonly group = new THREE.Group();
  private mat = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0, depthWrite: false, depthTest: false, toneMapped: false });
  private fillMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0, depthWrite: false, depthTest: false, side: THREE.DoubleSide, toneMapped: false });

  constructor(rig: RigJson, range = 4.2) {
    this.group.name = "frustums";
    for (const [name, c] of Object.entries(rig.cameras)) {
      const M = c.T_ego_cam;
      const R = new THREE.Matrix3().set(M[0][0], M[0][1], M[0][2], M[1][0], M[1][1], M[1][2], M[2][0], M[2][1], M[2][2]);
      const o = new THREE.Vector3(M[0][3], M[1][3], M[2][3]);
      const fx = c.params[0], fy = c.params[1] ?? fx, cx = c.params[2] ?? c.width / 2, cy = c.params[3] ?? c.height / 2;
      const fish = c.model === "OPENCV_FISHEYE";
      const ray = (u: number, v: number) => {
        const x = (u - cx) / fx, y = (v - cy) / fy;
        let d: THREE.Vector3;
        if (fish) {
          const th = Math.min(Math.hypot(x, y), Math.PI * 0.62);
          const ph = Math.atan2(y, x);
          d = new THREE.Vector3(Math.sin(th) * Math.cos(ph), Math.sin(th) * Math.sin(ph), Math.cos(th));
        } else d = new THREE.Vector3(x, y, 1).normalize();
        return d.applyMatrix3(R).multiplyScalar(range).add(o);
      };
      const border: THREE.Vector3[] = [];
      const N = 10;
      for (let i = 0; i < N; i++) border.push(ray((c.width * i) / N, 0));
      for (let i = 0; i < N; i++) border.push(ray(c.width, (c.height * i) / N));
      for (let i = N; i > 0; i--) border.push(ray((c.width * i) / N, c.height));
      for (let i = N; i > 0; i--) border.push(ray(0, (c.height * i) / N));
      const pts: number[] = [];
      for (let i = 0; i < border.length; i++) {
        const a = border[i], b = border[(i + 1) % border.length];
        pts.push(a.x, a.y, a.z, b.x, b.y, b.z);
      }
      for (const k of [0, N, 2 * N, 3 * N]) pts.push(o.x, o.y, o.z, border[k].x, border[k].y, border[k].z);
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
      const lines = new THREE.LineSegments(g, this.mat);
      lines.name = "frustum-" + name;
      lines.renderOrder = 30;
      lines.frustumCulled = false;
      this.group.add(lines);
      // faint fill of the far cap
      const fan: number[] = [];
      const ctr = ray(cx, cy);
      for (let i = 0; i < border.length; i++) {
        const a = border[i], b = border[(i + 1) % border.length];
        fan.push(ctr.x, ctr.y, ctr.z, a.x, a.y, a.z, b.x, b.y, b.z);
      }
      const fg = new THREE.BufferGeometry();
      fg.setAttribute("position", new THREE.Float32BufferAttribute(fan, 3));
      const fill = new THREE.Mesh(fg, this.fillMat);
      fill.renderOrder = 29;
      fill.frustumCulled = false;
      this.group.add(fill);
    }
  }

  setOpacity(v: number) {
    this.mat.opacity = 0.4 * v;
    this.fillMat.opacity = 0.05 * v;
    this.group.visible = v > 0.01;
  }
}
