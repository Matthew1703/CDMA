import queue
import threading
import time
from typing import Dict, List
import numpy as np

class CDMAController:
    """
    Класс 1: Контроллер CDMA системы
    - Генерирует коды Уолша
    - Управляет каналом связи
    - Суммирует все сигналы
    - Координирует работу
    """

    def __init__(self, num_stations: int = 4, code_length: int = 8, noise_level: float = 0.1):
            self.num_stations = num_stations
            self.code_length = code_length
            self.noise_level = noise_level
            
            self.station_words = {
                'A': "GOD",
                'B': "CAT",
                'C': "HAM",
                'D': "SUN"
            }
            
            self.walsh_codes = self.generate_walsh_codes()
            
            self.broadcast_queue = queue.Queue()
            
            self.tx_queue = queue.Queue()
            
            self.running = False
        
            self.sync_event = threading.Event()
            
            self.transmitters = []
            self.receivers = []
            
            self.transmission_history = []

    def generate_walsh_codes(self) -> Dict[str, np.ndarray]:
        """Генерация кодов Уолша для станций"""
        H = np.array([[1]])
        
        while H.shape[0] < self.code_length:
            H = np.vstack([
                np.hstack([H, H]),
                np.hstack([H, -H])
            ])
        
        codes = {}
        stations = list(self.station_words.keys())[:self.num_stations]
        
        for i, station_id in enumerate(stations):
            codes[station_id] = H[i]
        
        print("Сгенерированы коды Уолша:")
        for station_id, code in codes.items():
            print(f"  Станция {station_id}: {code}")
        
        return codes
    
    def start(self, duration: int = 20):
        """Запуск системы"""
        print("=" * 70)
        print("ЗАПУСК CDMA СИСТЕМЫ")
        print("=" * 70)
        
        self.running = True
        
        controller_thread = threading.Thread(target=self.controller_loop)
        controller_thread.start()
        
        print("\nЗапуск передатчиков...")
        for station_id, word in self.station_words.items():
            if station_id in self.walsh_codes:
                transmitter = Transmitter(
                    station_id=station_id,
                    word=word,
                    walsh_code=self.walsh_codes[station_id],
                    tx_queue=self.tx_queue,
                    controller=self
                )
                self.transmitters.append(transmitter)
                transmitter.start()
                time.sleep(0.2)
        
        print("\nЗапуск приемников...")
        for station_id in self.station_words.keys():
            if station_id in self.walsh_codes:
                receiver = Receiver(
                    receiver_id=f"RX_{station_id}",
                    station_id=station_id,
                    walsh_code=self.walsh_codes[station_id],
                    broadcast_queue=self.broadcast_queue,
                    controller=self
                )
                self.receivers.append(receiver)
                receiver.start()
                time.sleep(0.2)
        
        print(f"\nСистема работает {duration} секунд...")
        time.sleep(duration)
        
        self.stop()
        
        controller_thread.join(timeout=2)

    def controller_loop(self):
        """Основной цикл контроллера (суммирование и передача)"""
        print("[CONTROLLER] Контроллер запущен")
        
        transmission_count = 0
        
        while self.running:
            time.sleep(1)  
            
            signals = {}
            
            while not self.tx_queue.empty():
                try:
                    tx_data = self.tx_queue.get_nowait()
                    station_id = tx_data['station_id']
                    signal = tx_data['signal']
                    signals[station_id] = signal
                except queue.Empty:
                    break
            
            if signals:
                transmission_count += 1
                
                combined_signal = self.combine_signals(signals)
                
                noisy_signal = self.add_noise(combined_signal)
                
                self.broadcast_queue.put({
                    'cycle': transmission_count,
                    'signal': noisy_signal,
                    'timestamp': time.time(),
                    'active_stations': list(signals.keys())
                })
                
                self.transmission_history.append({
                    'cycle': transmission_count,
                    'signals': {k: v.tolist() for k, v in signals.items()},
                    'combined_signal': combined_signal.tolist(),
                    'noisy_signal': noisy_signal.tolist(),
                    'active_stations': list(signals.keys()),
                    'timestamp': time.time()
                })
                
                print(f"\n[CONTROLLER] Передача #{transmission_count}")
                print(f"  Активные станции: {list(signals.keys())}")
                print(f"  Длина суммарного сигнала: {len(combined_signal)}")
                if transmission_count == 1:
                    print(f"  Пример суммирования:")
                    for station_id, signal in signals.items():
                        print(f"    Станция {station_id}: {signal[:8]}...")
                    print(f"    Суммарный: {combined_signal[:8]}...")
        
        print("[CONTROLLER] Контроллер остановлен")

    def combine_signals(self, signals: Dict[str, np.ndarray]) -> np.ndarray:
        """Суммирование всех сигналов"""
        max_len = 0
        for signal in signals.values():
            max_len = max(max_len, len(signal))
        
        combined = np.zeros(max_len, dtype=np.float64)
        
        for signal in signals.values():
            if len(signal) < max_len:
                padded = np.pad(signal, (0, max_len - len(signal)))
                combined += padded
            else:
                combined += signal
        
        return combined

    def add_noise(self, signal: np.ndarray) -> np.ndarray:
        """Добавление шума к сигналу"""
        noise = np.random.normal(0, self.noise_level, len(signal))
        return signal + noise
    
    def stop(self):
        """Остановка системы"""
        print("\n" + "=" * 70)
        print("ОСТАНОВКА СИСТЕМЫ")
        print("=" * 70)
        
        self.running = False
        
        print("\nОстановка передатчиков...")
        for transmitter in self.transmitters:
            transmitter.running = False
        
        print("Остановка приемников...")
        for receiver in self.receivers:
            receiver.running = False
        
        time.sleep(1)
        
        self.print_statistics()
        self.print_transmission_history()

    def print_statistics(self):
        """Вывод статистики"""
        print("\n" + "=" * 70)
        print("СТАТИСТИКА РАБОТЫ")
        print("=" * 70)
        print(f"\nВсего циклов передачи: {len(self.transmission_history)}")
        
        if self.transmission_history:
            print("\nПример передачи (цикл 1):")
            last_tx = self.transmission_history[0]
            print(f"  Станций: {last_tx['active_stations']}")
            print(f"  Мощность суммарного сигнала: {np.mean(np.array(last_tx['combined_signal'])**2):.4f}")
            print(f"  Отношение сигнал/шум: {self.noise_level}")
        
        print("\nРабота приемников:")
        for receiver in self.receivers:
            print(f"  {receiver.receiver_id}: принято {receiver.received_count} сообщений")
            
    def print_transmission_history(self):
        """Вывод полной истории передач"""
        print("\n" + "=" * 70)
        print("ПОЛНАЯ ИСТОРИЯ ПЕРЕДАЧ")
        print("=" * 70)
        
        if not self.transmission_history:
            print("  История передач пуста")
            return
        
        print(f"\nВсего записей в истории: {len(self.transmission_history)}")
        
        for i, tx in enumerate(self.transmission_history):
            print(f"\n--- Передача #{i+1} (Цикл {tx['cycle']}) ---")
            print(f"  Время: {time.ctime(tx['timestamp'])}")
            print(f"  Активные станции: {tx['active_stations']}")
            print(f"  Длина сигнала: {len(tx['combined_signal'])} элементов")
            
            combined_arr = np.array(tx['combined_signal'])
            noisy_arr = np.array(tx['noisy_signal'])
            
            print(f"  Мощность чистого сигнала: {np.mean(combined_arr**2):.4f}")
            print(f"  Мощность сигнала с шумом: {np.mean(noisy_arr**2):.4f}")
            print(f"  Уровень шума: {np.mean((noisy_arr - combined_arr)**2):.4f}")
            
            print(f"  Пример сигнала (первые 10 элементов):")
            print(f"    Чистый:   {combined_arr[:10]}")
            print(f"    С шумом:  {noisy_arr[:10]}")
            
            if 'signals' in tx and tx['signals']:
                print(f"  Сигналы станций:")
                for station, signal_list in tx['signals'].items():
                    signal_arr = np.array(signal_list)
                    power = np.mean(signal_arr**2)
                    print(f"    Станция {station}: мощность={power:.4f}, длина={len(signal_arr)}")

class Transmitter(threading.Thread):
    """
    Класс 2: Передатчик (Базовая станция)
    - Кодирует данные с помощью кода Уолша
    - Отправляет сигнал контроллеру
    """
    def __init__(self, station_id: str, word: str, walsh_code: np.ndarray,
                    tx_queue: queue.Queue, controller: CDMAController):
            threading.Thread.__init__(self)
            self.station_id = station_id
            self.word = word
            self.walsh_code = walsh_code
            self.tx_queue = tx_queue
            self.controller = controller
            self.running = True
            self.tx_count = 0

    def run(self):
        """Основной цикл передатчика"""
        print(f"[TX {self.station_id}] Запущен. Слово: '{self.word}'")
        
        bits = self.text_to_bits(self.word)
        
        encoded_signal = self.encode_signal(bits)
        
        while self.running:  
            self.tx_queue.put({
                'station_id': self.station_id,
                'signal': encoded_signal,
                'tx_count': self.tx_count,
                'timestamp': time.time()
            })
            
            self.tx_count += 1
            print(f"[TX {self.station_id}] Отправлен сигнал #{self.tx_count}")
            
            time.sleep(1.5)
        
        print(f"[TX {self.station_id}] Остановлен. Всего передач: {self.tx_count}")

    def text_to_bits(self, text: str) -> List[int]:
        """Утилита: текст → биты ASCII"""
        bits = []
        for char in text:
            bits.extend([int(b) for b in format(ord(char), '08b')])
        return bits

    def encode_signal(self, bits: List[int]) -> np.ndarray:
        """Кодирование битов с использованием кода Уолша"""
        encoded_signal = np.array([], dtype=np.float64)
        
        for bit in bits:
            symbol = 1 if bit == 1 else -1  
            encoded_symbol = symbol * self.walsh_code
            encoded_signal = np.concatenate([encoded_signal, encoded_symbol])
        
        return encoded_signal

class Receiver(threading.Thread):
    """
    Класс 3: Приемник
    - Получает суммарный сигнал от контроллера
    - Декодирует с помощью кода Уолша
    - Извлекает свое сообщение
    """

    def __init__(self, receiver_id: str, station_id: str, walsh_code: np.ndarray,
                 broadcast_queue: queue.Queue, controller: CDMAController):
        threading.Thread.__init__(self)
        self.receiver_id = receiver_id
        self.station_id = station_id
        self.walsh_code = walsh_code
        self.broadcast_queue = broadcast_queue
        self.controller = controller
        self.running = True
        self.received_count = 0

    def run(self):
        """Основной цикл приемника"""
        print(f"[{self.receiver_id}] Запущен. Слушаю станцию {self.station_id}")
        
        while self.running:
            try:
                broadcast_data = self.broadcast_queue.get(timeout=1)
                
                combined_signal = broadcast_data['signal']
                cycle = broadcast_data['cycle']
                
                decoded_text = self.decode_signal(combined_signal)
                
                self.received_count += 1
                
                expected = self.controller.station_words[self.station_id]
                status = "OK" if decoded_text == expected else "FAILED"
                
                print(f"[{self.receiver_id}] Цикл {cycle}: '{decoded_text}' {status}")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[{self.receiver_id}] Ошибка: {e}")
                continue
        
        print(f"[{self.receiver_id}] Остановлен. Принято: {self.received_count}")
    
    def decode_signal(self, combined_signal: np.ndarray) -> str:
        """Декодирование суммарного сигнала"""
        decoded_bits = []
        code_length = len(self.walsh_code)
        
        num_blocks = len(combined_signal) // code_length
        
        for block in range(num_blocks):
            start = block * code_length
            end = start + code_length
            
            segment = combined_signal[start:end]
            
            correlation = np.sum(segment * self.walsh_code)
            
            bit = 1 if correlation > 0 else 0
            decoded_bits.append(bit)
        
        return self.bits_to_text(decoded_bits)
    
    def bits_to_text(self, bits: List[int]) -> str:
        """Утилита: биты → текст"""
        text = ""
        for i in range(0, len(bits), 8):
            if i + 8 > len(bits):
                break
            byte = bits[i:i+8]
            char_code = int(''.join(str(b) for b in byte), 2)
            text += chr(char_code)
        return text


def main():
    """Запускаем CDMA систему"""
    print("=" * 80)
    print("ПРОГРАММНАЯ МОДЕЛЬ CDMA СИСТЕМЫ")
    print("=" * 80)
    print("  - 4 базовые станции")
    print("  - 8-символьные коды Уолша с генерацией")
    print("  - Станции передают:")
    print("    A: 'GOD'")
    print("    B: 'CAT'")  
    print("    C: 'HAM'")
    print("    D: 'SUN'")
    print("=" * 80)
    
    controller = CDMAController(
        num_stations=4,
        code_length=8,
        noise_level=0  
    )
    
    try:
        controller.start(duration=30)
        
    except KeyboardInterrupt:
        print("\n\nПрограмма прервана пользователем")
        controller.stop()
    
    print("\nПрограмма завершена.")


if __name__ == "__main__":
    main()