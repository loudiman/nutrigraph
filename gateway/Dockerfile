# The gateway service. Node runs the TypeScript sources directly, so there is
# no build step and no compiled output to keep in step with the sources.
FROM node:24-slim

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci --omit=dev

COPY src ./src

CMD ["node", "src/index.ts"]
