# The gateway service. Node runs the TypeScript sources directly, so there is
# no build step and no compiled output to keep in step with the sources.
FROM node:24-slim

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci --omit=dev

COPY src ./src

# The minimal test client, and with it the demo-data warning issue #30 requires
# a User to see before they type. Without this line `express.static` serves a
# directory that is not in the image, and every path 404s on the deployed
# service while working perfectly on a laptop.
COPY public ./public

CMD ["node", "src/index.ts"]
