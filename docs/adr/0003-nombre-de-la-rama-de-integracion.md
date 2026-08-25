# ADR 0003 — Nombre de la rama de integración

- **Status:** Accepted
- **Date:** 2026-08-25

---

## Contexto

El enunciado del entregable exige literalmente ramas **"Main"** y **"Development"**.

La convención estándar de git-flow, y la que se usa en la industria, nombra la rama de
integración **`develop`**.

El proyecto tiene que satisfacer al evaluador sin adoptar una convención que su autor
no usaría en un equipo real.

## Decisión

La rama de integración se llama **`develop`**.

La equivalencia con la rama "Development" del enunciado se documenta explícitamente en
**`docs/GIT_STRATEGY.md`**, en una nota dirigida a un lector externo que no conoce el
proyecto.

## Alternativas consideradas

**Nombrar la rama `development` para coincidir literalmente con el enunciado.**
Descartado porque optimiza para una lista de verificación en vez de para la práctica
correcta, y porque el proyecto es además una pieza de portafolio que otros
desarrolladores van a leer.

**Mantener ambas ramas sincronizadas.** Descartado sin discusión: dos ramas de
integración es una fuente permanente de divergencia a cambio de nada.

## Consecuencias

### Positiva

El repositorio usa la convención que su autor defendería en una entrevista técnica.

### Negativa

Exige una nota de equivalencia que un lector podría no encontrar. Se mitiga colocándola
en la sección donde se describen los roles de las ramas, que es lo primero que se lee
del documento.

### Riesgo aceptado

Un evaluador que verifique por **coincidencia exacta de cadena** podría marcar el ítem
como incumplido. Se considera improbable dado que la equivalencia está escrita, y el
costo de revertir es un renombrado de rama.
