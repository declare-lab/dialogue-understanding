# MultiWOZ 2.2

This dataset is consisted of a schema file `schema.json` describing the ontology
and dialogue files `dialogues_*.json` of dialogue data under `train`, `dev`, and
`test` folders.

## Schema file

`schema.json` defines the new ontology using the schema representation in
[schema-guided dialogue dataset](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue#scheme-representation]).

Table below shows the categorical slots, non-categorical slots and intents
defined for each domain.

| Domain     | Categorical slots       | Non-categorical slots   | Intents    |
| ---------- | :---------------------: | :---------------------: | :--------: |
| Restaurant | pricerange, area,       | food, name, booktime,   | find, book |
:            : bookday, bookpeople     : address, phone,         :            :
:            :                         : postcode, ref           :            :
| Attraction | area, type              | name, address,          | find       |
:            :                         : entrancefee, openhours  :            :
:            :                         : phone, postcode,        :            :
| Hotel      | pricerange, parking,    | name, address, phone,   | find, book |
:            : internet, stars, area,  : postcode, ref           :            :
:            : type, bookpeople        :                         :            :
:            : ,bookday, bookstay      :                         :            :
| Taxi       | -                       | destination, departure, | book       |
:            :                         : arriveby, leaveat,      :            :
:            :                         : phone, type             :            :
| Train      | destination, departure, | arriveby, leaveat,      | find, book |
:            : day, bookpeople         : trainid, ref, price,    :            :
:            :                         : duration                :            :
| Bus        | day                     | departure, destination, | find       |
:            :                         : leaveat                 :            :
| Hospital   | -                       | department , address,   | find       |
:            :                         : phone, postcode         :            :
| Police     | -                       | name, address, phone,   | find       |
:            :                         : postcode                :            :

Among all the 61 slots in the schema, 35 slots are tracked in the dialogue state
as listed below.

```json
{
  'attraction-area',
  'attraction-name',
  'attraction-type',
  'bus-day',
  'bus-departure',
  'bus-destination',
  'bus-leaveat',
  'hospital-department',
  'hotel-area',
  'hotel-bookday',
  'hotel-bookpeople',
  'hotel-bookstay',
  'hotel-internet',
  'hotel-name',
  'hotel-parking',
  'hotel-pricerange',
  'hotel-stars',
  'hotel-type',
  'restaurant-area',
  'restaurant-bookday',
  'restaurant-bookpeople',
  'restaurant-booktime',
  'restaurant-food',
  'restaurant-name',
  'restaurant-pricerange',
  'taxi-arriveby',
  'taxi-departure',
  'taxi-destination',
  'taxi-leaveat',
  'train-arriveby',
  'train-bookpeople',
  'train-day',
  'train-departure',
  'train-destination',
  'train-leaveat'
}
```

## Dialogue files

Dialogues are formatted following the data presentation of
[schema-guided dialogue dataset](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue#dialogue-representation).
The state of a slot is presented as a list of values. Predicting any of them is
considered as correct. In addition, we also add the span annotations that
identifies the location where slot values have been mentioned in the utterances
for non-categorical slots. Its annotation is presented as,

```json
{
  "slots": [
    {
      "slot": a string of slot name.
      "start": The index of the starting character in the utterance corresponding to the slot value.
      "exclusive_end": The index of the character just after the last character corresponding to the slot value in the utterance. In python, utterance[start:exclusive_end] gives the slot value.
      "value": a string of value.
    }
  ]
}
```

There are some non-categorical slots whose values are carried over from another
slot in the dialogue state. Their values don't explicitly appear in the
utterances. For these slots, instead of annotating the spans, we use "copy from"
annotation to identify the slot it copies the value from. Its annotation is
formatted as follows,

```json
{
  "slots": [
    {
      "slot": a string of slot name
      "copy_from": the slot to copy from.
      "value": a string of value.
    }
  ]
}
```

## Action annotation

We observed some wrong and incomplete action annotations in MultiWOZ 2.1. We
estimate to release the corrected action annotations around the end of July,
2020.

## Conversion to the data format of MultiWOZ 2.1

To include the corrections from MultiWOZ 2.2 dataset into MultiWOZ 2.1 in the
format used by the MultiWOZ 2.1 dataset, please download the
[MultiWOZ 2.1](https://github.com/budzianowski/multiwoz/blob/master/data/MultiWOZ_2.1.zip)
zip file, unzip it, and run

```bash
python convert_to_multiwoz_format.py --multiwoz21_data_dir=<multiwoz21_data_dir> --output_file=<output json file>
```

Please refer to our
[paper](https://www.aclweb.org/anthology/2020.nlp4convai-1.13.pdf) for more
details about the dataset.
