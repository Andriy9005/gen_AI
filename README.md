# gen_AI
UK_EN_translator
CAPSTONE GEN AI
Ідея проєкту: 
  Створення генеративної моделі повного циклу, яку б можна було використати наприклад при надиктовці голосового тексту в парі EN_UKR та видавала голосовий переклад 
відповідно. 
Логіка реалізації: 
  Послідовне застосування наступних етапів: 
   a) використання STT з перетворення голосу в текст (з автодетекцією на вході мови мовлення) 
   б) пошук створеної LLM/MT, найбільш підхожої для цілей створення моделей перекладу, пошук прийнятного дата-сету, файн-тюнінг моделі для перекладу 
      в обидві сторони, оцінка результатів, подальше застосування 
   в) використання TTS з перетворення тексту в голос (з автодетекцією на вході мови мовлення) 
   г) отримання та оцінювання результатів BLEU 
   д) (що було б ідеально встигнути зробити) адаптувати код для flask та створення простої html - сторінки/додатку 
   е) наступні кроки по вдосконаленню: донавчити модель EN_UK для більш якісного контекстного перекладу, вдосконалити логіку для запровадження синхронності 
     (оптимізація коду, пришвидшення виконання, аби очікування перекладу було менше ніж зараз).

Перелік релевантних файлів/папок на Github: 

  -> __pycache__  - кешовані файли токенайзера при виконанні коду
  -> bleu_results - автоматизовані (1000 тест на прикладів) результати навчання LLM за метрикою BLEU
  -> examples_tests - wav файли тестування фінального коду з перекладом в обидві сторони
  -> fine_tuning_auto_validation - папка зі скриптами по файн-тюнінгу LLM
  -> speech_2_voice_ai - папка зі скриптами і фіналізованим кодом інтегрованої моделі перекладу 
        ->  	dictophone_trio_3.py  - фінальний код моделі
  -> statiс - наразі пуста папка длі веб інтегрованого коду
  -> templates - папка для веб - інтеграції index.html
  -> trained_models - лінки на скачування натренованих LLM/MT моделей, які перекладають текст
  -> training_inputs - скрипти для обєднання дата-сету для тренування LLM/MT, лінки на джерело відкритого дата-сету 
  Файли: 
  readme.txt - короткий огляд та структура матеріалів 
  CAPSTONE GEN AI.pptx - звіт про виконаний проєкт, логіка/фази
  tokenizer.py - бекап-версі токенайзера для української озвучки
