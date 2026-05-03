<template>
  <div
    class="relative"
    @mouseenter="open = true"
    @mouseleave="open = false"
    @click="open = !open"
  >
    <button class="nav-link font-semibold flex items-center gap-1 cursor-pointer">
      {{ title }}
      <span class="text-xs transition-transform" :class="open ? 'rotate-180' : ''">▼</span>
    </button>
    <div
      v-show="open"
      class="absolute top-full left-0 mt-1 bg-white border rounded shadow-lg py-1 min-w-[200px] z-50"
    >
      <template v-for="item in items" :key="item.label">
        <router-link
          v-if="!item.external"
          :to="item.to"
          class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
          @click="open = false"
        >
          {{ item.label }}
        </router-link>
        <a
          v-else
          :href="item.href"
          target="_blank"
          rel="noopener noreferrer"
          class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
        >
          {{ item.label }}
        </a>
      </template>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

defineProps({
  title: String,
  items: Array
})

const open = ref(false)
</script>
