<template>
  <div>
    <h1 class="text-2xl font-bold text-gray-900 font-serif">Algorithm Training</h1>
    <div class="mt-4 space-y-8 text-gray-700 font-serif leading-relaxed text-justify">

      <section>
        <h2 class="text-xl font-semibold border-b border-gray-100 pb-1">C++ Notes</h2>
        <ul class="mt-2 space-y-1 text-sm list-disc list-inside">
          <li>scanf does not support string (C++ class)</li>
          <li>Mathematical expressions and algorithm complexity analysis</li>
        </ul>
      </section>

      <section v-for="group in acwingBlocks" :key="group.title">
        <h2 class="text-xl font-semibold border-b border-gray-100 pb-1">{{ group.title }}</h2>
        <div class="mt-3 bg-gray-900 rounded-lg overflow-hidden shadow-md">
          <div class="flex items-center justify-between px-4 py-2 bg-gray-800 border-b border-gray-700">
            <span class="text-xs font-mono text-gray-400">C++</span>
            <button
              @click="toggleCode(group.title)"
              class="text-xs text-gray-400 hover:text-gray-200 transition-colors"
            >
              {{ expandedCodes[group.title] ? 'Collapse' : 'Expand' }}
            </button>
          </div>
          <pre
            v-show="expandedCodes[group.title]"
            class="p-4 text-sm font-mono text-gray-200 overflow-x-auto"
            style="white-space: pre; tab-size: 2;"
          ><code>{{ group.code }}</code></pre>
        </div>
      </section>

      <section>
        <h2 class="text-xl font-semibold border-b border-gray-100 pb-1">Data Structures & Algorithms</h2>
        
        <div v-for="group in algoBlocks" :key="group.title" class="mt-4">
          <h3 class="text-lg font-semibold">{{ group.title }}</h3>
          <div class="mt-2 bg-gray-900 rounded-lg overflow-hidden shadow-md">
            <div class="flex items-center justify-between px-4 py-2 bg-gray-800 border-b border-gray-700">
              <span class="text-xs font-mono text-gray-400">C++</span>
              <button
                @click="toggleCode(group.title)"
                class="text-xs text-gray-400 hover:text-gray-200 transition-colors"
              >
                {{ expandedCodes[group.title] ? 'Collapse' : 'Expand' }}
              </button>
            </div>
            <pre
              v-show="expandedCodes[group.title]"
              class="p-4 text-sm font-mono text-gray-200 overflow-x-auto"
              style="white-space: pre; tab-size: 2;"
            ><code>{{ group.code }}</code></pre>
          </div>
        </div>
      </section>

      <section>
        <h2 class="text-xl font-semibold border-b border-gray-100 pb-1">LeetCode Practice</h2>
        
        <div v-for="group in leetcodeBlocks" :key="group.title" class="mt-4">
          <h3 class="text-lg font-semibold text-blue-800">{{ group.title }}</h3>
          <div class="mt-2 bg-gray-900 rounded-lg overflow-hidden shadow-md">
            <div class="flex items-center justify-between px-4 py-2 bg-gray-800 border-b border-gray-700">
              <span class="text-xs font-mono text-gray-400">C++</span>
              <button
                @click="toggleCode(group.title)"
                class="text-xs text-gray-400 hover:text-gray-200 transition-colors"
              >
                {{ expandedCodes[group.title] ? 'Collapse' : 'Expand' }}
              </button>
            </div>
            <pre
              v-show="expandedCodes[group.title]"
              class="p-4 text-sm font-mono text-gray-200 overflow-x-auto"
              style="white-space: pre; tab-size: 2;"
            ><code>{{ group.code }}</code></pre>
          </div>
        </div>
      </section>

      <section>
        <h2 class="text-xl font-semibold border-b border-gray-100 pb-1">Core Competencies</h2>

        <h3 class="text-lg font-semibold mt-4">Machine Learning Fundamentals</h3>
        <ul class="mt-2 space-y-1 list-disc list-inside text-sm">
          <li>Supervised & Unsupervised Learning</li>
          <li>Classification & Regression</li>
          <li>Feature Engineering & Selection</li>
          <li>Model Evaluation & Validation</li>
        </ul>

        <h3 class="text-lg font-semibold mt-4">Deep Learning</h3>
        <ul class="mt-2 space-y-1 list-disc list-inside text-sm">
          <li>Neural Network Architectures</li>
          <li>Convolutional Neural Networks (CNNs)</li>
          <li>Recurrent Neural Networks (RNNs)</li>
          <li>Transformer Architecture</li>
        </ul>

        <h3 class="text-lg font-semibold mt-4">Large Language Models</h3>
        <ul class="mt-2 space-y-1 list-disc list-inside text-sm">
          <li>Pre-training & Fine-tuning Techniques</li>
          <li>Supervised Fine-Tuning (SFT)</li>
          <li>Reinforcement Learning from Human Feedback (RLHF)</li>
          <li>Agent Development & Harness</li>
        </ul>

        <h3 class="text-lg font-semibold mt-4">Recommendation Systems</h3>
        <ul class="mt-2 space-y-1 list-disc list-inside text-sm">
          <li>Collaborative Filtering</li>
          <li>Content-Based Filtering</li>
          <li>Deep Learning based Recommendations</li>
          <li>Real-time Recommendation Systems</li>
        </ul>
      </section>

      <section>
        <h2 class="text-xl font-semibold border-b border-gray-100 pb-1">Key Technologies</h2>
        <div class="mt-4 grid grid-cols-2 md:grid-cols-4 gap-3">
          <div class="bg-gray-50 rounded-lg p-3 text-center">
            <span class="text-sm font-medium text-gray-800">C++</span>
          </div>
          <div class="bg-gray-50 rounded-lg p-3 text-center">
            <span class="text-sm font-medium text-gray-800">Python</span>
          </div>
          <div class="bg-gray-50 rounded-lg p-3 text-center">
            <span class="text-sm font-medium text-gray-800">PyTorch</span>
          </div>
          <div class="bg-gray-50 rounded-lg p-3 text-center">
            <span class="text-sm font-medium text-gray-800">TensorFlow</span>
          </div>
          <div class="bg-gray-50 rounded-lg p-3 text-center">
            <span class="text-sm font-medium text-gray-800">CUDA</span>
          </div>
          <div class="bg-gray-50 rounded-lg p-3 text-center">
            <span class="text-sm font-medium text-gray-800">Hugging Face</span>
          </div>
          <div class="bg-gray-50 rounded-lg p-3 text-center">
            <span class="text-sm font-medium text-gray-800">FAISS</span>
          </div>
          <div class="bg-gray-50 rounded-lg p-3 text-center">
            <span class="text-sm font-medium text-gray-800">PySpark</span>
          </div>
        </div>
      </section>

      <section>
        <h2 class="text-xl font-semibold border-b border-gray-100 pb-1">Technical Summary</h2>
        <p class="mt-2">
          Comprehensive notes on algorithms, data structures, and machine learning. 
          Applied across various domains including robotics, recommendation systems, 
          and natural language processing.
        </p>
      </section>

    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import codeblocks from '@/data/codeblocks.json'

const expandedCodes = ref({})

const toggleCode = (title) => {
  expandedCodes.value[title] = !expandedCodes.value[title]
}

const acwingBlocks = computed(() => 
  codeblocks.filter(b => b.title.startsWith('AcWing') || /^\d+\./.test(b.title))
)

const algoBlocks = computed(() =>
  codeblocks.filter(b => 
    b.title.includes('基础算法') || 
    b.title.includes('习题课') || 
    b.title.includes('数据结构')
  )
)

const leetcodeBlocks = computed(() =>
  codeblocks.filter(b => 
    b.title.includes('Two Sum') || 
    b.title.includes('Leetcode')
  )
)
</script>
