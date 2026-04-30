ARG BASE_IMAGE=nvcr.io/nvidia/isaac-sim
FROM ${BASE_IMAGE}

# Arguments to match host user
ARG USER_NAME=isaac_developer
ARG UID=1000
ARG GID=1000
ARG WORK_DIR=/isaac-sim

# Switch to root to perform setup
USER root

# Install sudo and basic tools
RUN apt-get update && apt-get install -y sudo git vim && \
    rm -rf /var/lib/apt/lists/*

# Setup user and group matching the host
# 1. Remove existing user/group if they use the same UID/GID to avoid conflicts
# 2. Create the new user
RUN if getent passwd ${UID}; then userdel -f $(getent passwd ${UID} | cut -d: -f1); fi && \
    if getent group ${GID}; then groupdel $(getent group ${GID} | cut -d: -f1); fi && \
    groupadd -g ${GID} ${USER_NAME} && \
    useradd -u ${UID} -g ${GID} -m -s /bin/bash ${USER_NAME} && \
    echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USER_NAME} && \
    chmod 0440 /etc/sudoers.d/${USER_NAME}

# --- PERMISSION FIXES ---

# 1. Isaac Sim Specifics
# Create writable directories for kit data/logs/cache
RUN mkdir -p /isaac-sim/kit/data /isaac-sim/kit/logs /isaac-sim/kit/cache && \
    chown -R ${UID}:${GID} /isaac-sim/kit/data /isaac-sim/kit/logs /isaac-sim/kit/cache && \
    chmod 775 /isaac-sim/kit/data /isaac-sim/kit/logs /isaac-sim/kit/cache && \
    # Ensure /isaac-sim is readable by others (the created user)
    if [ -d "/isaac-sim" ]; then chmod o+rx /isaac-sim; fi

# 2. Isaac Lab Specifics (Critical step for Lab image)
# If /workspace/isaaclab exists (Isaac Lab Image), transfer ownership to non-root user
# so python can write bytecodes and logs.
RUN if [ -d "/workspace/isaaclab" ]; then \
        chown -R ${UID}:${GID} /workspace/isaaclab; \
    else \
        # If it doesn't exist (Sim Image), just make the dir for potential mounts
        mkdir -p /workspace/isaaclab && chown ${UID}:${GID} /workspace/isaaclab; \
    fi

# 3. Create standard cache directories to ensure user owns them
RUN mkdir -p /home/${USER_NAME}/.cache/ov \
    /home/${USER_NAME}/.cache/pip \
    /home/${USER_NAME}/.cache/nvidia/GLCache \
    /home/${USER_NAME}/.nv/ComputeCache \
    /home/${USER_NAME}/.nvidia-omniverse/logs \
    /home/${USER_NAME}/.local/share/ov/data \
    && chown -R ${UID}:${GID} /home/${USER_NAME}

# Copy entrypoint script and make it executable.
# Build context is sim/docker/ — see docker-compose.yml.
COPY entrypoint.sh /usr/local/bin/isaac_entrypoint.sh
RUN chmod +x /usr/local/bin/isaac_entrypoint.sh

# Switch to the new user
USER ${USER_NAME}
ENV HOME=/home/${USER_NAME}

# Set the working directory
WORKDIR ${WORK_DIR}

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/isaac_entrypoint.sh"]