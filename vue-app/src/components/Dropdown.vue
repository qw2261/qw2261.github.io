<template>
  <div
    class="relative"
    @mouseenter="open = true"
    @mouseleave="open = false"
  >
    <button
      class="nav-link font-semibold flex items-center gap-1 cursor-pointer"
      @click.stop="open = !open"
    >
      {{ title }}
      <span class="text-xs transition-transform duration-200" :class="open ? 'rotate-180' : ''">▼</span>
    </button>
    <div
      v-show="open"
      class="absolute top-full left-0 pt-2"
    >
      <div class="bg-white border border-gray-200 rounded-lg shadow-lg py-1 min-w-[220px] z-50">
        <template v-for="item in items" :key="item.label">
          <router-link
            v-if="!item.external"
            :to="item.to"
            class="block px-4 py-2.5 text-sm text-gray-700 hover:bg-gray-50 hover:text-gray-900 transition-colors"
            @click="open = false"
          >
            {{ item.label }}
          </router-link>
          <a
            v-else
            :href="item.href"
            target="_blank"
            rel="noopener noreferrer"
            class="block px-4 py-2.5 text-sm text-gray-700 hover:bg-gray-50 hover:text-gray-900 transition-colors"
            @click="open = false"
          >
            {{ item.label }}
          </a>
        </template>
      </div>
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
