import React, { useEffect } from 'react'
import * as THREE from 'three'

// Cube component: creates a mesh and adds to provided scene.
// Props:
//  - scene: THREE.Scene
//  - position: [x,y,z]
//  - size: number
//  - color: hex or CSS string
//  - name: identifier
//  - onCreate(mesh) optional

export default function Cube({ scene, position = [0,0,0], size = 1, color = '#888', name, onCreate }) {
  useEffect(() => {
    if (!scene) return
    const geometry = new THREE.BoxGeometry(size, size, size)
    const material = new THREE.MeshStandardMaterial({ color })
    const mesh = new THREE.Mesh(geometry, material)
    mesh.position.set(position[0], position[1], position[2])
    mesh.userData.name = name
    scene.add(mesh)
    if (onCreate) onCreate(mesh)
    return () => {
      scene.remove(mesh)
      geometry.dispose()
      material.dispose()
    }
  }, [scene, position, size, color])

  return null
}
