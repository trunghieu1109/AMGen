async def forward_160(self, taskInfo):
    logs = []
    loop_results = {
        'stage_0.subtask_1': {'thinking': [], 'answer': []},
        'stage_0.subtask_2': {'thinking': [], 'answer': []},
        'stage_0.subtask_3': {'thinking': [], 'answer': []},
        'stage_1.subtask_1': {'thinking': [], 'answer': []},
        'stage_1.subtask_2': {'thinking': [], 'answer': []}
    }

    for iteration in range(2):
        cot_instruction_0_1 = (
            "Sub-task 1: Extract and summarize all given information from the query relevant to the mean free path problem, "
            "ensuring to capture the scenario details and parameters accurately. Input: taskInfo containing the question and choices."
        )
        cot_agent_desc_0_1 = {
            'instruction': cot_instruction_0_1,
            'input': [taskInfo],
            'temperature': 0.0,
            'context': ['user query']
        }
        results_0_1, log_0_1 = await self.cot(subtask_id='stage_0.subtask_1', cot_agent_desc=cot_agent_desc_0_1)
        loop_results['stage_0.subtask_1']['thinking'].append(results_0_1['thinking'])
        loop_results['stage_0.subtask_1']['answer'].append(results_0_1['answer'])
        logs.append(log_0_1)

        cot_instruction_0_2 = (
            "Sub-task 2: Explicitly define and distinguish the two mean free paths λ1 and λ2, specifying their physical meanings, units, "
            "and typical representative formulas or values for gas–gas collision mean free path (λ1) and electron–gas scattering mean free path (λ2) at 1000 kV accelerating voltage. "
            "This subtask addresses the previous failure to clarify what λ2 represents and prevents contradictory assumptions downstream. "
            "Input: taskInfo, thinking and answer from stage_0.subtask_1."
        )
        cot_agent_desc_0_2 = {
            'instruction': cot_instruction_0_2,
            'input': [taskInfo] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_1']['answer'],
            'temperature': 0.0,
            'context': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1']
        }
        results_0_2, log_0_2 = await self.cot(subtask_id='stage_0.subtask_2', cot_agent_desc=cot_agent_desc_0_2)
        loop_results['stage_0.subtask_2']['thinking'].append(results_0_2['thinking'])
        loop_results['stage_0.subtask_2']['answer'].append(results_0_2['answer'])
        logs.append(log_0_2)

        cot_instruction_0_3 = (
            "Sub-task 3: Identify and categorize key physical parameters and variables affecting λ1 and λ2, including pressure, temperature, volume, gas molecule density, "
            "and electron beam parameters, ensuring to incorporate the clarified definitions from stage_0.subtask_2. "
            "This subtask ensures all relevant factors are considered with clear variable roles. "
            "Input: taskInfo, thinking and answer from stage_0.subtask_2."
        )
        cot_agent_desc_0_3 = {
            'instruction': cot_instruction_0_3,
            'input': [taskInfo] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'],
            'temperature': 0.0,
            'context': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2']
        }
        results_0_3, log_0_3 = await self.cot(subtask_id='stage_0.subtask_3', cot_agent_desc=cot_agent_desc_0_3)
        loop_results['stage_0.subtask_3']['thinking'].append(results_0_3['thinking'])
        loop_results['stage_0.subtask_3']['answer'].append(results_0_3['answer'])
        logs.append(log_0_3)

        cot_instruction_1_1 = (
            "Sub-task 1: Analyze the relationship between electron beam initiation and changes in mean free path, focusing on electron–gas molecule scattering effects "
            "and how they modify the effective mean free path from λ1 to λ2. Use the definitions and parameters clarified in stage_0 to avoid ambiguity about which scattering cross sections apply. "
            "Input: taskInfo, thinking and answer from stage_0.subtask_3."
        )
        cot_agent_desc_1_1 = {
            'instruction': cot_instruction_1_1,
            'input': [taskInfo] + loop_results['stage_0.subtask_3']['thinking'] + loop_results['stage_0.subtask_3']['answer'],
            'temperature': 0.0,
            'context': ['user query', 'thinking of stage_0.subtask_3', 'answer of stage_0.subtask_3']
        }
        results_1_1, log_1_1 = await self.cot(subtask_id='stage_1.subtask_1', cot_agent_desc=cot_agent_desc_1_1)
        loop_results['stage_1.subtask_1']['thinking'].append(results_1_1['thinking'])
        loop_results['stage_1.subtask_1']['answer'].append(results_1_1['answer'])
        logs.append(log_1_1)

        cot_instruction_1_2 = (
            "Sub-task 2: Derive or justify mathematically any factor relating λ2 to λ1 (e.g., the previously mentioned 1.22 multiplier) based on the ratio of electron–gas and gas–gas scattering cross sections at 1000 kV. "
            "This subtask addresses the prior unjustified use of the 1.22 factor by providing a clear derivation or rejecting it if unsupported. "
            "Input: taskInfo, thinking and answer from stage_1.subtask_1."
        )
        cot_agent_desc_1_2 = {
            'instruction': cot_instruction_1_2,
            'input': [taskInfo] + loop_results['stage_1.subtask_1']['thinking'] + loop_results['stage_1.subtask_1']['answer'],
            'temperature': 0.0,
            'context': ['user query', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1']
        }
        results_1_2, log_1_2 = await self.cot(subtask_id='stage_1.subtask_2', cot_agent_desc=cot_agent_desc_1_2)
        loop_results['stage_1.subtask_2']['thinking'].append(results_1_2['thinking'])
        loop_results['stage_1.subtask_2']['answer'].append(results_1_2['answer'])
        logs.append(log_1_2)

    cot_instruction_2_1 = (
        "Sub-task 1: Based on the analysis of scattering effects and the mathematical derivation of any scaling factors, derive a clear conclusion about the relationship between λ2 and λ1, "
        "explicitly stating which inequality or equality holds and why. This subtask integrates all prior reasoning and definitions to produce a logically supported conclusion. "
        "Input: taskInfo, all thinking and answers from stage_0.subtask_3, stage_1.subtask_1, and stage_1.subtask_2."
    )
    cot_agent_desc_2_1 = {
        'instruction': cot_instruction_2_1,
        'input': [taskInfo] + loop_results['stage_0.subtask_3']['thinking'] + loop_results['stage_0.subtask_3']['answer'] + loop_results['stage_1.subtask_1']['thinking'] + loop_results['stage_1.subtask_1']['answer'] + loop_results['stage_1.subtask_2']['thinking'] + loop_results['stage_1.subtask_2']['answer'],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_0.subtask_3', 'answer of stage_0.subtask_3', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1', 'thinking of stage_1.subtask_2', 'answer of stage_1.subtask_2']
    }
    results_2_1, log_2_1 = await self.cot(subtask_id='stage_2.subtask_1', cot_agent_desc=cot_agent_desc_2_1)
    logs.append(log_2_1)

    cot_agent_instruction_3_1 = (
        "Sub-task 1: Format the final conclusion clearly and select the correct choice about λ2 from the given options, ensuring the answer is concise and directly supported by the prior derivation and analysis. "
        "Input: taskInfo, thinking and answer from stage_2.subtask_1."
    )
    cot_agent_desc_3_1 = {
        'instruction': cot_agent_instruction_3_1,
        'input': [taskInfo, results_2_1['thinking'], results_2_1['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_2.subtask_1', 'answer of stage_2.subtask_1']
    }
    results_3_1, log_3_1 = await self.answer_generate(subtask_id='stage_3.subtask_1', cot_agent_desc=cot_agent_desc_3_1)
    logs.append(log_3_1)

    final_answer = await self.make_final_answer(results_3_1['thinking'], results_3_1['answer'])
    return final_answer, logs
