import {
  vec3,
  mat4,
} from './wgpu-matrix.module.min.js';

const adapter = await navigator.gpu.requestAdapter();
const limits = adapter.limits;
const device = await adapter.requestDevice({ requiredFeatures: ['timestamp-query'], requiredLimits: {
	maxBufferSize: limits.maxBufferSize,
	maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
}
});

const canvas = document.getElementById("canvas");
const context = canvas.getContext("webgpu");
const format = navigator.gpu.getPreferredCanvasFormat();

context.configure({
	device,
	format,
	alphaMode: "opaque"
});

const states = 8;
const threshold = 15;
const range = 2;
const gridSize = 256;
const gridSizeSq = gridSize**2;


// N 8, t 15, r 2, size: 256, moore
// N 40, t 1, r 1, size: 512, neumann

const workgroupX = 8;
const workgroupY = 4;
const workgroupZ = 2;
const usePackedValues = true;
const vonNeumannNeighborhood = false;

const startRecording = false;
const recordingResolution = 1024;

function bitWidth(n) {
  let width = 1;
  n = n >> 1;
  while (n > 0) {
    width += 1;
    n = n >> 1;
  }
  return width;
}
function nextPowerOfTwo(n) {
	if (n == 0) return 1;
	n -= 1;
	n |= n >> 1;
	n |= n >> 2;
	n |= n >> 4;
	n |= n >> 8;
	n |= n >> 16;
	n += 1;
	return n;
}

const packedValueSize = nextPowerOfTwo(bitWidth(states - 1));
const packedValueCount = 32 / packedValueSize;
const packedValueMask = (1 << packedValueSize) - 1;
const gridSizeXDimension = gridSize / packedValueCount;

function downloadCanvasPNG(namePrefix = "canvas") {
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const filename = `${namePrefix}_${stamp}.png`;

  canvas.toBlob((blob) => {
    if (!blob) return;
    const a = document.createElement("a");
    a.download = filename;
    a.href = URL.createObjectURL(blob);
    a.click();
    URL.revokeObjectURL(a.href);
  }, "image/png");
}

const shaderModule = device.createShaderModule({
	code:
`
/*
enable subgroups;
*/

struct VSOut {
	@builtin(position) position: vec4f,
	
	@location(0) cube_pos: vec3f,
	@location(1) world_pos: vec3f
};

struct Constants {
	mvp: mat4x4<f32>,
	eye_pos: vec3<f32>,
	time: f32,
	padding0: f32,
	padding1: f32,
	padding2: f32,
};

@group(0) @binding(0) var<uniform> constants : Constants;

@group(1) @binding(0) var<storage, read> readBuffer : array<u32>;
@group(1) @binding(1) var<storage, read_write> writeBuffer : array<u32>;

fn hsl(h: f32, s: f32, l: f32) -> vec3f {
    let c = (1.0 - abs(2.0 * l - 1.0)) * s;
    let h_ = h * 6.0; // sector 0..6
    let x = c * (1.0 - abs(h_ % 2.0 - 1.0));

    var rgb = vec3f(0.0);

    if (h_ < 1.0) {
        rgb = vec3f(c, x, 0.0);
    } else if (h_ < 2.0) {
        rgb = vec3f(x, c, 0.0);
    } else if (h_ < 3.0) {
        rgb = vec3f(0.0, c, x);
    } else if (h_ < 4.0) {
        rgb = vec3f(0.0, x, c);
    } else if (h_ < 5.0) {
        rgb = vec3f(x, 0.0, c);
    } else {
        rgb = vec3f(c, 0.0, x);
    }

    let m = l - 0.5 * c;
    return rgb + vec3f(m);
}

const gridSize = ${gridSize};
const gridSizeSq = ${gridSizeSq};

const packedValueSize = ${packedValueSize};
const packedValueCount = ${packedValueCount};
const packedValueMask = ${packedValueMask};
const gridSizeXDimension = ${gridSizeXDimension};

const roseSandPalette = array<vec3<f32>, 10>(
    vec3<f32>(0.298, 0.000, 0.059), // deep wine rose
    vec3<f32>(0.525, 0.063, 0.145), // rich rose red
    vec3<f32>(0.741, 0.251, 0.275), // coral rose
    vec3<f32>(0.878, 0.447, 0.408), // soft clay
    vec3<f32>(0.949, 0.643, 0.510), // warm terracotta
    vec3<f32>(0.925, 0.745, 0.588), // light sand
    vec3<f32>(0.909, 0.800, 0.682), // beige tan
    vec3<f32>(0.949, 0.870, 0.745), // pale cream
    vec3<f32>(0.976, 0.925, 0.847), // warm ivory
    vec3<f32>(0.988, 0.957, 0.909)  // off-white highlight
);

const coralReef = array<vec3<f32>, 16>(
    vec3f(0.941, 0.506, 0.396), // coral (#f08165)
    vec3f(0.851, 0.408, 0.345), // red coral
    vec3f(0.851, 0.298, 0.282), // reef red (#d95348)
    vec3f(0.890, 0.596, 0.431), // sandy orange
    vec3f(0.929, 0.765, 0.549), // pale sand
    vec3f(0.812, 0.863, 0.725), // seafoam
    vec3f(0.600, 0.855, 0.804), // turquoise light
    vec3f(0.341, 0.741, 0.737), // cyan reef (#24aebb)
    vec3f(0.239, 0.588, 0.624), // deep teal
    vec3f(0.188, 0.435, 0.529), // ocean shadow
    vec3f(0.231, 0.396, 0.569), // deep blue violet
    vec3f(0.361, 0.435, 0.647), // periwinkle
    vec3f(0.569, 0.506, 0.682), // soft lavender coral
    vec3f(0.729, 0.545, 0.655), // shell pink
    vec3f(0.824, 0.529, 0.553), // rose coral
    vec3f(0.941, 0.506, 0.396)  // wrap back to coral
);

@vertex
fn vs_main(@builtin(instance_index) instance_id: u32, @location(0) in_pos: vec3f) -> VSOut {
	let pos = in_pos;

	var out: VSOut;
	out.position = constants.mvp * vec4f(pos, 1.0);
	out.cube_pos = (pos + 0.5) * (2047.0 / 2048.0);
	out.world_pos = pos;
	return out;
}

// Source: https://www.shadertoy.com/view/XtGGzG
fn magma_quintic(x_in: f32) -> vec3<f32> {
    let x  = clamp(x_in, 0.0, 1.0);
    let x1 = vec4<f32>(1.0, x, x * x, x*x*x);
    let x2 = x1 * x1.w * x;

    let r = dot(x1, vec4<f32>(-0.023226960,  1.087154378, -0.109964741,  6.333665763))
          + dot(x2.xy, vec2<f32>(-11.640596589,  5.337625354));

    let g = dot(x1, vec4<f32>( 0.010680993,  0.176613780,  1.638227448, -6.743522237))
          + dot(x2.xy, vec2<f32>( 11.426396979, -5.523236379));

    let b = dot(x1, vec4<f32>(-0.008260782,  2.244286052,  3.005587601, -24.279769818))
          + dot(x2.xy, vec2<f32>( 32.484310068, -12.688259703));

    return saturate(vec3f(r, g, b));
}

fn viridis_quintic(x_in: f32) -> vec3<f32> {
    let x  = saturate(x_in);
    let x1 = vec4<f32>(1.0, x, x * x, x * x * x);  // 1, x, x^2, x^3
    let x2 = x1 * x1.w * x;                        // x^4, x^5, x^6, x^7

    let r = dot(x1, vec4<f32>( 0.280268003, -0.143510503,  2.225793877, -14.815088879))
          + dot(x2.xy, vec2<f32>(25.212752309, -11.772589584));

    let g = dot(x1, vec4<f32>(-0.002117546,  1.617109353, -1.909305070,   2.701152864))
          + dot(x2.xy, vec2<f32>(-1.685288385,  0.178738871));

    let b = dot(x1, vec4<f32>( 0.300805501,  2.614650302, -12.019139090,  28.933559110))
          + dot(x2.xy, vec2<f32>(-33.491294770, 13.762053843));

    return saturate(vec3<f32>(r, g, b));
}

fn plasma_quintic(x_in: f32) -> vec3<f32> {
    let x  = saturate(x_in);
    let x1 = vec4<f32>(1.0, x, x * x, x * x * x);  // 1, x, x^2, x^3
    let x2 = x1 * x1.w * x;                        // x^4, x^5, x^6, x^7

    let r = dot(x1, vec4<f32>( 0.063861086,  1.992659096, -1.023901152, -0.490832805))
          + dot(x2.xy, vec2<f32>( 1.308442123, -0.914547012));

    let g = dot(x1, vec4<f32>( 0.049718590, -0.791144343,  2.892305078,  0.811726816))
          + dot(x2.xy, vec2<f32>(-4.686502417,  2.717794514));

    let b = dot(x1, vec4<f32>( 0.513275779,  1.580255060, -5.164414457,  4.559573646))
          + dot(x2.xy, vec2<f32>(-1.916810682,  0.570638854));

    return saturate(vec3<f32>(r, g, b));
}


@fragment
fn fs_main(@location(0) cube_pos: vec3f, @location(1) world_pos: vec3f) -> @location(0) vec4f {

	const drawOutline = false;
	if (drawOutline) {
		let edge = abs(cube_pos - 0.5) > vec3f(0.5 - (1.0 / 164.0));
		if (all(edge.xy) || all(edge.xz) || all(edge.yz)) {
				return vec4f(vec3f(0.0, 1.0, 0.0), 1.0);
		}
	}

	var rayOrigin = vec3f(cube_pos * vec3f(gridSize));

	let rayDirection = normalize(world_pos - constants.eye_pos);
	// let rayDirection = normalize(world_pos - vec3f(sin(0.7854) * 1.5, sin(0.7854) * 1.5, cos(0.7854) * 1.5));
	const center = vec3f(gridSize) / 2.0;

	const renderAsSphere = false;
	if (renderAsSphere) {
		let sphereCenter = center - rayOrigin;
		let t = dot(sphereCenter, rayDirection);
		let projectedPoint = rayDirection * t;

		const radius = center.x;
		let distanceFromCenter = length(sphereCenter - projectedPoint);

		if (distanceFromCenter > radius+1.0) {
			return vec4f(vec3f(0.0, 0.0, 0.0), 1.0);
		}

	}

	var rayPosition = rayOrigin;

	let delta = abs(1.0 / rayDirection);
	let step = sign(rayDirection);

	let f = fract(rayPosition);
	var sideDistance = vec3f(
		select(f.x, 1.0 - f.x, rayDirection.x >= 0) * delta.x,
		select(f.y, 1.0 - f.y, rayDirection.y >= 0) * delta.y,
		select(f.z, 1.0 - f.z, rayDirection.z >= 0) * delta.z
	);

	var voxel_state : u32;
	loop {
		if (usePackedValues) {
			let flatIndex = u32(rayPosition.z)*(gridSizeXDimension*gridSize) + u32(rayPosition.y)*(gridSizeXDimension) + (u32(rayPosition.x) / packedValueCount);
			let shiftAmount = (u32(rayPosition.x) % packedValueCount) * packedValueSize;
			voxel_state = (readBuffer[flatIndex] >> shiftAmount) & packedValueMask;
		} else {
			let flatIndex = u32(rayPosition.z)*gridSizeSq + u32(rayPosition.y)*gridSize + u32(rayPosition.x);
			voxel_state = readBuffer[flatIndex];
		}

		var isTransparent = voxel_state <= 5;
		if (renderAsSphere) {
			let isInCenter = distance(center, ceil(rayPosition)) < center.x;
			if (!isInCenter) {
				isTransparent = true;
				rayOrigin = rayPosition;
			}
		}
		if (!isTransparent) {
			break;
		}

		let m = min(min(sideDistance.x, sideDistance.y), sideDistance.z);
		if (m == sideDistance.x) {
			sideDistance.x += delta.x;
			rayPosition.x += step.x;
		} else if (m == sideDistance.y) {
			sideDistance.y += delta.y;
			rayPosition.y += step.y;
		} else {
			sideDistance.z += delta.z;
			rayPosition.z += step.z;
		}

		if (any(rayPosition >= vec3f(gridSize)) || any(rayPosition < vec3f(0.0))) {
			return vec4f(vec3f(0.0), 1.0);
		}
	}

	let distance = length(rayPosition - rayOrigin);
	// let fade = exp(-distance * (1.0 / 32.0));
	let fade = exp(-distance * (1.0 / 64.0));

	// let c = hsl(f32(voxel_state) / f32(states) * 0.5 + 0.3, 0.65, 0.5);
	// let c = vec3f(f32(voxel_state) / f32(states)) * fade;
	// let c = normalize(world_pos - constants.eye_pos);
	let c = coralReef[voxel_state];
	// let c = magma_quintic(f32(voxel_state) / f32(states));
	return vec4f(c * fade, 1.0);
}


const usePackedValues = ${usePackedValues ? "true" : "false"};
const vonNeumannNeighborhood = ${vonNeumannNeighborhood ? "true" : "false"};
const states = ${states};
const threshold = ${threshold};
const range = ${range};

fn wrapCoord(n: i32) -> i32 {
	if (n >= (gridSize)) { return n - gridSize; };
	if (n < 0) { return n + gridSize; };
	return n;
}

fn wrapCoordPacked(n: i32) -> i32 {
	if (n >= (gridSizeXDimension)) { return n - gridSizeXDimension; };
	if (n < 0) { return n + gridSizeXDimension; };
	return n;
}

fn wrapCoords(coords: vec3i) -> vec3i {
	return vec3i(wrapCoord(coords.x), wrapCoord(coords.y), wrapCoord(coords.z));
}

fn cca_basic(id: vec3<u32>) {
	let idx = id.z*gridSizeSq + id.y*gridSize + id.x;
	let currentState = readBuffer[idx];
	let nextState = select(currentState + 1, 0, currentState + 1 == states);
	var count = 0u;
	if (vonNeumannNeighborhood) {
		for (var z = -range; z <= range; z += 1) {
			let nz = wrapCoord(z + i32(id.z)) * gridSizeSq;
			let ySearch = range - abs(z);
			for (var y = -ySearch; y <= ySearch; y += 1) {
				let ny = wrapCoord(y + i32(id.y)) * gridSize;
				let xSearch = range - abs(y) - abs(z);
				for (var x = -xSearch; x <= xSearch; x += 1) {
					if (z == 0 && y == 0 && x == 0) { continue; }
					let nx = wrapCoord(x + i32(id.x));
					if (readBuffer[nz + ny + nx] == nextState) {
						count += 1u;
					}
				}
			}
		}
	} else {
		for (var z = -range; z <= range; z += 1) {
			let nz = wrapCoord(z + i32(id.z)) * gridSizeSq;
			for (var y = -range; y <= range; y += 1) {
				let ny = wrapCoord(y + i32(id.y)) * gridSize;
				for (var x = -range; x <= range; x += 1) {
					if (z == 0 && y == 0 && x == 0) { continue; }
					let nx = wrapCoord(x + i32(id.x));
					if (readBuffer[nz + ny + nx] == nextState) {
						count += 1u;
					}
				}
			}
		}
	}
	writeBuffer[idx] = select(currentState, nextState, count >= threshold);
}

fn flatten_index(index: vec3i, dimensions: vec3i) -> i32 {
	return (index.z*(dimensions.x*dimensions.y)) + index.y*dimensions.x + index.x;
}

fn cca_packed(id: vec3<u32>) {
	let idx = id.z * (gridSize * gridSizeXDimension) + id.y * gridSizeXDimension + id.x;
	let packedStates = readBuffer[idx];

	var currentStates = array<u32, packedValueCount>();
	var nextStates = array<u32, packedValueCount>();

	for (var i = 0u; i < packedValueCount; i += 1) {
		let state = (packedStates >> (i * packedValueSize)) & packedValueMask;
		let nextState = select(state + 1, 0, state + 1 == states);
		currentStates[i] = state;
		nextStates[i] = nextState;
	}

	var counts = array<u32, packedValueCount>();

	if (vonNeumannNeighborhood) {
		for (var z = -range; z <= range; z += 1) {
			let nz = wrapCoord(z + i32(id.z)) * (gridSize * gridSizeXDimension);
			let yRange = range - abs(z);
			for (var y = -yRange; y <= yRange; y += 1) {
				let ny = wrapCoord(y + i32(id.y)) * gridSizeXDimension;

				const maxLength = 1 + ((range + (packedValueCount - 1)) / packedValueCount) * 2;
				var adjacentStates = array<u32, maxLength>();
				let xRange = range - abs(y) - abs(z);
				let leftmostX = i32(id.x) - ((xRange + packedValueCount - 1) / packedValueCount);
				let len = 1 + ((xRange + (packedValueCount - 1)) / packedValueCount) * 2;
				let offset = (packedValueCount - (xRange % packedValueCount)) % packedValueCount;

				for (var i = 0; i < len; i += 1) {
					adjacentStates[i] = readBuffer[nz + ny + wrapCoordPacked(leftmostX + i)];
				}

				for (var i = 0u; i < packedValueCount; i += 1) {
					var c = 0u;
					let nextState = nextStates[i];

					for (var x = 0u; x <= u32(xRange * 2); x += 1) {
						let localX = x + i + u32(offset);
						let states = adjacentStates[localX / packedValueCount];
						let shiftAmount = (localX % packedValueCount) * packedValueSize;
						let state = u32(states >> shiftAmount) & packedValueMask;
						c += u32(state == nextState);
					}
					counts[i] += c;
				}
			}
		}
	} else {
		let leftmostX = i32(id.x) - ((range + packedValueCount - 1) / packedValueCount);
		for (var z = -range; z <= range; z += 1) {
			let nz = wrapCoord(z + i32(id.z)) * (gridSize * gridSizeXDimension);
			for (var y = -range; y <= range; y += 1) {
				let ny = wrapCoord(y + i32(id.y)) * gridSizeXDimension;

				const len = 1 + ((range + (packedValueCount - 1)) / packedValueCount) * 2;
				var packedWords = array<u32, len>();
				for (var i = 0; i < len; i += 1) {
					packedWords[i] = readBuffer[nz + ny + wrapCoordPacked(leftmostX + i)];
				}

				for (var i = 0u; i < packedValueCount; i += 1) {
					var c = 0u;
					let nextState = nextStates[i];

					for (var x = 0u; x <= range * 2; x += 1) {
						const offset = (packedValueCount - (range % packedValueCount)) % packedValueCount;
						let localX = x + i + offset;
						let states = packedWords[localX / packedValueCount];
						let shiftAmount = (localX % packedValueCount) * packedValueSize;
						let state = u32(states >> shiftAmount) & packedValueMask;
						c += u32(state == nextState);
					}
					counts[i] += c;
				}
			}
		}
	}

	var out = 0u;
	for (var i = 0u; i < packedValueCount; i += 1) {
		let newState = select(currentStates[i], nextStates[i], counts[i] >= threshold);
		out |= newState << (i * packedValueSize);
	}
	writeBuffer[idx] = out;
}

@compute
@workgroup_size(${workgroupX}, ${workgroupY}, ${workgroupZ})
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
	if (usePackedValues) {
		cca_packed(id);
	} else {
		cca_basic(id);
	}
}
`
});

const vertices = new Float32Array([
  -0.5, -0.5,  0.5,
   0.5, -0.5,  0.5,
   0.5,  0.5,  0.5,
  -0.5, -0.5,  0.5,
   0.5,  0.5,  0.5,
  -0.5,  0.5,  0.5,

   0.5, -0.5, -0.5,
  -0.5, -0.5, -0.5,
  -0.5,  0.5, -0.5,
   0.5, -0.5, -0.5,
  -0.5,  0.5, -0.5,
   0.5,  0.5, -0.5,

   0.5, -0.5,  0.5,
   0.5, -0.5, -0.5,
   0.5,  0.5, -0.5,
   0.5, -0.5,  0.5,
   0.5,  0.5, -0.5,
   0.5,  0.5,  0.5,

  -0.5, -0.5, -0.5,
  -0.5, -0.5,  0.5,
  -0.5,  0.5,  0.5,
  -0.5, -0.5, -0.5,
  -0.5,  0.5,  0.5,
  -0.5,  0.5, -0.5,

  -0.5,  0.5,  0.5,
   0.5,  0.5,  0.5,
   0.5,  0.5, -0.5,
  -0.5,  0.5,  0.5,
   0.5,  0.5, -0.5,
  -0.5,  0.5, -0.5,

  -0.5, -0.5, -0.5,
   0.5, -0.5, -0.5,
   0.5, -0.5,  0.5,
  -0.5, -0.5, -0.5,
   0.5, -0.5,  0.5,
  -0.5, -0.5,  0.5,
]);
const vertexBuffer = device.createBuffer({
	mappedAtCreation: true,
	size: vertices.byteLength,
	usage: GPUBufferUsage.VERTEX
});

{
	const bufferHandle = new Float32Array(vertexBuffer.getMappedRange());
	bufferHandle.set(vertices);
	vertexBuffer.unmap();
}

let bufferIndex = 0;

const uniformBuffer = device.createBuffer({
	size: 256,
	usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
});

const bufferSize = gridSize**2 * (usePackedValues ? gridSizeXDimension : gridSize) * 4;
const buffers = [
	device.createBuffer({
		mappedAtCreation: true,
		size: bufferSize,
		usage: GPUBufferUsage.STORAGE,
	}),
	device.createBuffer({
		size: bufferSize,
		usage: GPUBufferUsage.STORAGE
	})
];

function mulberry32(seed) {
	let t = seed;
	return function() {
		t |= 0;
		t = t + 0x6D2B79F5 | 0;
		let r = Math.imul(t ^ t >>> 15, 1 | t);
		r = r + Math.imul(r ^ r >>> 7, 61 | r) ^ r;
		return ((r ^ r >>> 14) >>> 0) / 4294967296;
	};
}

const deterministicRandomness = true;
const rand = (deterministicRandomness) ? mulberry32(12345) : Math.random;

{
	const initialBufferData = new Uint32Array(buffers[0].getMappedRange());
	if (usePackedValues) {
		for (let i = 0; i < bufferSize; ++i) {
			let r = 0;
			for (let j = 0; j < packedValueCount; ++j) {
				r |= Math.floor(rand() * states) << (j * packedValueSize);
				// r |= Math.floor(j % states) << (j * packedValueSize);
			}
			initialBufferData[i] = r;
		}
	} else {
		for (let i = 0; i < gridSize**3; ++i) {
			initialBufferData[i] = rand() * states;
		}
	}
	buffers[0].unmap();
}

let bindGroupIndex = 0;

const group0 = device.createBindGroupLayout({
	entries: [
		{
			binding: 0,
			visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
			buffer: { type: "uniform" }
		}
	]
});

const group1Render = device.createBindGroupLayout({
	entries: [
		{
			binding: 0,
			visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
			buffer: { type: "read-only-storage" }
		},
	]
});
const group1Compute = device.createBindGroupLayout({
	entries: [
		{
			binding: 0,
			visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
			buffer: { type: "read-only-storage" }
		},
		{
			binding: 1,
			visibility: GPUShaderStage.COMPUTE,
			buffer: { type: "storage" }
		}
	]
});

const renderPipeline = device.createRenderPipeline({
	layout: device.createPipelineLayout({ bindGroupLayouts: [group0, group1Render] }),
	vertex: {
		entryPoint: "vs_main",
		module: shaderModule,
		buffers: [{
			arrayStride: 3 * 4,
			attributes: [
				{ shaderLocation: 0, offset: 0, format: "float32x3" }
			]
		}]
	},
	fragment: {
		entryPoint: "fs_main",
		module: shaderModule,
		targets: [{ format }]
	},
	primitive: {
		topology: "triangle-list",
		frontFace: 'ccw',
		cullMode: 'back'
	},
});

const computePipeline = device.createComputePipeline({
	layout: device.createPipelineLayout({ bindGroupLayouts: [group0, group1Compute] }),
	compute: {
		entryPoint: "cs_main",
		module: shaderModule
	}
});

const renderUniformBindGroup = device.createBindGroup({
	layout: group0,
	entries: [
		{ binding: 0, resource: { buffer: uniformBuffer } }
	]
});

const renderBindGroups = [
	device.createBindGroup({
		layout: group1Render,
		entries: [
			{ binding: 0, resource: { buffer: buffers[0] } },
		]
	}),
	device.createBindGroup({
		layout: group1Render,
		entries: [
			{ binding: 0, resource: { buffer: buffers[1] } },
		]
	})
];

const computeBindGroups = [
	device.createBindGroup({
		layout: group1Compute,
		entries: [
			{ binding: 0, resource: { buffer: buffers[0] } },
			{ binding: 1, resource: { buffer: buffers[1] } },
		]
	}),
	device.createBindGroup({
		layout: group1Compute,
		entries: [
			{ binding: 0, resource: { buffer: buffers[1] } },
			{ binding: 1, resource: { buffer: buffers[0] } },
		]
	})
];

const querySet = device.createQuerySet({ type: 'timestamp', count: 4 });
const queryBuffer = device.createBuffer({
	size: 256,
	usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC
});
const queryBufferCPU = device.createBuffer({
	size: 256,
	usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
});

const start = document.timeline.currentTime;
let lastTime = start;

const frameSkip = 3;
let frameIndex = -1;
const useBenchmarks = true;

const samples = 128;
const times = new Float64Array(samples);
let timesIndex = 0;
let sufficientData = false;

let writingBenchmarkResults = false;
async function writeResults(frameIndex) {
	writingBenchmarkResults = true;
	await queryBufferCPU.mapAsync(GPUMapMode.READ);
	const view = new BigInt64Array(queryBufferCPU.getMappedRange());
	const renderPassDuration = view[1] - view[0];
	const computePassDuration = view[3] - view[2];
	queryBufferCPU.unmap();
	writingBenchmarkResults = false;

	const granularity = 1000;
	const renderPassDurationMS = Number(renderPassDuration * BigInt(granularity) / BigInt(1e6)) / granularity;
	const computePassDurationMS = Number(computePassDuration * BigInt(granularity) / BigInt(1e6)) / granularity;
	times[timesIndex] = computePassDurationMS;
	timesIndex = (timesIndex + 1) % samples;
	if (timesIndex == 0) sufficientData = true;

	if (sufficientData) {
		const sorted_times = new Float64Array(samples);
		sorted_times.set(times, 0);
		sorted_times.sort((a, b) => a - b); // ascending

		const median = sorted_times[Math.floor(samples / 2)].toFixed(2);
		const worst = sorted_times[samples - 1];
		const best = sorted_times[0];
		const mean = sorted_times.reduce((a, b) => a + b) / samples;
		let variance = 0;
		for (let i = 0; i < samples; ++i) {
			variance += (sorted_times[i] - mean)**2;
		}
		variance /= samples;
		const std_dev = Math.sqrt(variance);
		// console.log(`Compute Pass: ${sorted_times[50].toFixed(2)}ms`);
		console.log(`Compute Pass\nMedian: ${median}\nWorst: ${worst.toFixed(2)}\nBest:${best.toFixed(2)}\nMean: ${mean.toFixed(2)}\nStandard Deviation: ${std_dev.toFixed(2)}`);
	}
	// console.log(`Frame ${frameIndex} - Render Pass: ${renderPassDurationMS}ms - Compute Pass: ${computePassDurationMS}ms`);
}

function degreesToRadians(degrees) {
	return degrees / 180.0 * Math.PI;
}

let runningSimulation = true;

let angleX = degreesToRadians(45);

let moveLeft = false;
let moveRight = false;

window.onkeyup = (e) => {
	if (e.key == " ") {
		runningSimulation = !runningSimulation;
	}
	if (e.key == "ArrowLeft" || e.key == "a") {
		moveLeft = false;
	}
	if (e.key == "ArrowRight" || e.key == "d") {
		moveRight = false;
	}
}
window.onkeydown = (e) => {
	if (e.key == "ArrowLeft" || e.key == "a") {
		moveLeft = true;
	}
	if (e.key == "ArrowRight" || e.key == "d") {
		moveRight = true;
	}
}

let recorder;

async function frame(currentTime) {

	frameIndex += 1;

	if (startRecording) {
		if (frameIndex == 3) {
			recorder = setupRecorder();
			recorder.start();
		} else if (frameIndex == 3 + 60 * 9) {
			recorder.stop();
		}
	}

	const time = (currentTime - start) * (1.0 / 1024.0);
	const dt = currentTime - lastTime;
	lastTime = currentTime;

	if (moveLeft) {
		angleX -= dt / 256.0;
	}
	if (moveRight) {
		angleX += dt / 256.0;
	}

	{
		const eye = vec3.create(Math.sin(angleX) * 1.5, Math.sin(degreesToRadians(40)) * 1.5, Math.cos(angleX) * 1.5);
		const target = vec3.create(0, -0.1, 0);
		const up = vec3.create(0, 1, 0);
		const view = mat4.lookAt(eye, target, up);
		const fovy = 60 * (Math.PI / 180);
		const proj = mat4.perspective(fovy, canvas.width / canvas.height, 1/8, 1024);
		const mvp = mat4.multiply(proj, view);
		const inverse_view = mat4.inverse(view);

		const upload = new Float32Array(16 + 4 + 4);
		upload.set(mvp, 0);
		upload.set(eye, 16);
		upload[20] = time;
		device.queue.writeBuffer(uniformBuffer, 0, upload);
	}

	const view = context.getCurrentTexture().createView();
	const encoder = device.createCommandEncoder();
	{
		const pass = encoder.beginRenderPass({
			colorAttachments: [{
				view,
				loadOp: "clear",
				storeOp: "store",
				clearValue: { r: 0.0, g: 0, b: 0, a: 1 }
			}],
			timestampWrites: {
				querySet,
				beginningOfPassWriteIndex: 0,
				endOfPassWriteIndex: 1,
			}
		});
		pass.setPipeline(renderPipeline);
		pass.setVertexBuffer(0, vertexBuffer);
		pass.setBindGroup(0, renderUniformBindGroup);
		pass.setBindGroup(1, renderBindGroups[bindGroupIndex]);
		pass.draw(36, 1);
		pass.end();
	}
	if (frameIndex % frameSkip == 0 && runningSimulation) {
		const pass = encoder.beginComputePass({
			timestampWrites: {
				querySet,
				beginningOfPassWriteIndex: 2,
				endOfPassWriteIndex: 3,
			}
		});
		pass.setPipeline(computePipeline);
		pass.setBindGroup(0, renderUniformBindGroup);
		pass.setBindGroup(1, computeBindGroups[bindGroupIndex]);
		if (usePackedValues) {
			pass.dispatchWorkgroups(gridSizeXDimension / workgroupX, gridSize / workgroupY, gridSize / workgroupZ);
		} else {
			pass.dispatchWorkgroups(gridSize / workgroupX, gridSize / workgroupY, gridSize / workgroupZ);
		}
		pass.end();
		bindGroupIndex ^= 1;

		if (useBenchmarks) {
			encoder.resolveQuerySet(querySet, 0, 4, queryBuffer, 0);
			if (!writingBenchmarkResults) {
				encoder.copyBufferToBuffer(queryBuffer, 0, queryBufferCPU, 0, 256);
			}
		}
	}

	device.queue.submit([encoder.finish()]);
	if (useBenchmarks && !writingBenchmarkResults) {
		writeResults(frameIndex);
	}

	requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

function resize() {
	if (startRecording && (canvas.width != recordingResolution || canvas.height != recordingResolution)) {
		canvas.width = canvas.height = recordingResolution;
		return;
	}
	const w = window.innerWidth;
	const h = window.innerHeight;
	if (canvas.width != w || canvas.height != h) {
		canvas.width = w;
		canvas.height = h;
	}
}
resize();
window.onresize = resize;

function setupRecorder() {
	const fps = 20;
	const stream = canvas.captureStream(fps);

	const mimeType =
	  MediaRecorder.isTypeSupported('video/webm;codecs=vp9') ? 'video/webm;codecs=vp9' :
	  MediaRecorder.isTypeSupported('video/webm;codecs=vp8') ? 'video/webm;codecs=vp8' :
	  'video/webm';

	const recorder = new MediaRecorder(stream, {
		mimeType,
		videoBitsPerSecond: 64_000_000,
	});

	const chunks = [];
	recorder.ondataavailable = e => e.data.size && chunks.push(e.data);
	recorder.onstop = () => {
		console.log("recorder stopped!");
		const blob = new Blob(chunks, { type: mimeType });
		const url = URL.createObjectURL(blob);
		const a = Object.assign(document.createElement('a'), { href: url, download: 'capture.webm' });
		a.click();
		URL.revokeObjectURL(url);
	};
	return recorder;
}
