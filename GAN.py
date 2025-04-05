# GAN.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Generator model (generates fake data)
def build_generator(latent_dim):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=latent_dim))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1000, activation='sigmoid'))  # Same shape as input data
    return model

# Discriminator model (classifies real vs fake)
def build_discriminator(input_dim):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_dim=input_dim))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Output is real or fake (0 or 1)
    return model

# GAN model combines both (generator + discriminator)
def build_gan(generator, discriminator):
    discriminator.trainable = False  # During GAN training, the discriminator is frozen
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

# Parameters
latent_dim = 100  # Latent space dimension (input to generator)
input_dim = 1000  # Same as the number of features in your dataset

# Build and compile models
generator = build_generator(latent_dim)
discriminator = build_discriminator(input_dim)
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

gan = build_gan(generator, discriminator)
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Train the GAN
def train_gan(generator, discriminator, gan, X_train, epochs=10000, batch_size=64):
    batch_count = X_train.shape[0] // batch_size

    for epoch in range(epochs):
        for _ in range(batch_count):
            # Train discriminator with real data
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_data = X_train[idx]

            # Generate fake data
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            fake_data = generator.predict(noise)

            # Train discriminator on real and fake data
            d_loss_real = discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))
            d_loss_fake = discriminator.train_on_batch(fake_data, np.zeros((batch_size, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train generator (fool the discriminator)
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

        # Print loss every 1000 epochs
        if epoch % 1000 == 0:
            print(f"Epoch {epoch} | D Loss: {d_loss[0]} | G Loss: {g_loss}")

# Train the GAN
train_gan(generator, discriminator, gan, X_train)