async def forward_183(self, taskInfo):
    logs = []
    loop_results = {
        'stage_0.subtask_1': {'thinking': [], 'answer': []},
        'stage_0.subtask_2': {'thinking': [], 'answer': []},
        'stage_0.subtask_3': {'thinking': [], 'answer': []},
        'stage_0.subtask_4': {'thinking': [], 'answer': []},
        'stage_0.subtask_5': {'thinking': [], 'answer': []}
    }

    for iteration in range(2):
        cot_instruction_0_1 = (
            "Sub-task 1: Analyze the target molecule 2-(tert-butyl)-1-ethoxy-3-nitrobenzene and starting material benzene to identify required substituents, their positions, and necessary reaction types, explicitly considering directing effects and compatibility to avoid early ring deactivation (e.g., do not start with nitration). "
            "Input: taskInfo"
        )
        cot_agent_desc_0_1 = {
            'instruction': cot_instruction_0_1,
            'input': [taskInfo],
            'temperature': 0.0,
            'context': ['user query']
        }
        results_0_1, log_0_1 = await self.cot(
            subtask_id='stage_0.subtask_1',
            cot_agent_desc=cot_agent_desc_0_1
        )
        loop_results['stage_0.subtask_1']['thinking'].append(results_0_1['thinking'])
        loop_results['stage_0.subtask_1']['answer'].append(results_0_1['answer'])
        logs.append(log_0_1)

        cot_instruction_0_2 = (
            "Sub-task 2: Map each required reaction type to the corresponding reagents and conditions from the given choices, ensuring reagent roles are clearly assigned and incompatible or redundant steps (e.g., unnecessary sulfonation) are identified and flagged. "
            "Input: taskInfo, all previous thinking and answers from stage_0.subtask_1 and all previous iterations of stage_0.subtask_5"
        )
        cot_agent_desc_0_2 = {
            'instruction': cot_instruction_0_2,
            'input': [taskInfo] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_1']['answer'] + loop_results['stage_0.subtask_5']['thinking'] + loop_results['stage_0.subtask_5']['answer'],
            'temperature': 0.0,
            'context': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1', 'thinking of previous stage_0.subtask_5 iterations', 'answer of previous stage_0.subtask_5 iterations']
        }
        results_0_2, log_0_2 = await self.cot(
            subtask_id='stage_0.subtask_2',
            cot_agent_desc=cot_agent_desc_0_2
        )
        loop_results['stage_0.subtask_2']['thinking'].append(results_0_2['thinking'])
        loop_results['stage_0.subtask_2']['answer'].append(results_0_2['answer'])
        logs.append(log_0_2)

        cot_instruction_0_3 = (
            "Sub-task 3: Generate candidate reaction sequences for each choice, strictly enforcing correct order of steps based on directing effects and reactivity (e.g., Friedel-Crafts alkylation must precede nitration; avoid multiple nitrations; minimize or exclude sulfonation unless justified). "
            "Input: taskInfo, all previous thinking and answers from stage_0.subtask_2 and all previous iterations of stage_0.subtask_4"
        )
        cot_agent_desc_0_3 = {
            'instruction': cot_instruction_0_3,
            'input': [taskInfo] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'] + loop_results['stage_0.subtask_4']['thinking'] + loop_results['stage_0.subtask_4']['answer'],
            'temperature': 0.0,
            'context': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2', 'thinking of previous stage_0.subtask_4 iterations', 'answer of previous stage_0.subtask_4 iterations']
        }
        results_0_3, log_0_3 = await self.cot(
            subtask_id='stage_0.subtask_3',
            cot_agent_desc=cot_agent_desc_0_3
        )
        loop_results['stage_0.subtask_3']['thinking'].append(results_0_3['thinking'])
        loop_results['stage_0.subtask_3']['answer'].append(results_0_3['answer'])
        logs.append(log_0_3)

        cot_instruction_0_4 = (
            "Sub-task 4: Document detailed reasoning for each candidate sequence, explicitly addressing potential issues such as ring deactivation, directing group effects, sulfonation roles, and compatibility of functional group transformations, incorporating failure reasons from previous evaluations to avoid repeating errors. "
            "Input: taskInfo, all previous thinking and answers from stage_0.subtask_3 and all previous iterations of stage_0.subtask_3"
        )
        cot_agent_desc_0_4 = {
            'instruction': cot_instruction_0_4,
            'input': [taskInfo] + loop_results['stage_0.subtask_3']['thinking'] + loop_results['stage_0.subtask_3']['answer'] + loop_results['stage_0.subtask_3']['thinking'] + loop_results['stage_0.subtask_3']['answer'],
            'temperature': 0.0,
            'context': ['user query', 'thinking of stage_0.subtask_3', 'answer of stage_0.subtask_3', 'thinking of previous stage_0.subtask_3 iterations', 'answer of previous stage_0.subtask_3 iterations']
        }
        results_0_4, log_0_4 = await self.cot(
            subtask_id='stage_0.subtask_4',
            cot_agent_desc=cot_agent_desc_0_4
        )
        loop_results['stage_0.subtask_4']['thinking'].append(results_0_4['thinking'])
        loop_results['stage_0.subtask_4']['answer'].append(results_0_4['answer'])
        logs.append(log_0_4)

        cot_agent_instruction_0_5 = (
            "Sub-task 5: Produce a structured summary of intermediate sequences with annotations on expected yields, regioselectivity, and chemical feasibility, to serve as a clear basis for refinement and selection in the next stage. "
            "Input: taskInfo, all previous thinking and answers from stage_0.subtask_4 and all previous iterations of stage_0.subtask_5"
        )
        cot_agent_desc_0_5 = {
            'instruction': cot_agent_instruction_0_5,
            'input': [taskInfo] + loop_results['stage_0.subtask_4']['thinking'] + loop_results['stage_0.subtask_4']['answer'] + loop_results['stage_0.subtask_5']['thinking'] + loop_results['stage_0.subtask_5']['answer'],
            'temperature': 0.0,
            'context': ['user query', 'thinking of stage_0.subtask_4', 'answer of stage_0.subtask_4', 'thinking of previous stage_0.subtask_5 iterations', 'answer of previous stage_0.subtask_5 iterations']
        }
        results_0_5, log_0_5 = await self.answer_generate(
            subtask_id='stage_0.subtask_5',
            cot_agent_desc=cot_agent_desc_0_5
        )
        loop_results['stage_0.subtask_5']['thinking'].append(results_0_5['thinking'])
        loop_results['stage_0.subtask_5']['answer'].append(results_0_5['answer'])
        logs.append(log_0_5)

    cot_reflect_instruction_1_1 = (
        "Sub-task 1: Simplify and consolidate the intermediate sequences generated in stage_0.subtask_5 by removing redundant, low-yield, or chemically infeasible steps, explicitly avoiding sequences that start with nitration or misuse sulfonation, as per expert feedback. "
        "Input: all thinking and answers from stage_0.subtask_5"
    )
    critic_instruction_1_1 = (
        "Please review and provide the limitations of provided solutions of stage_0.subtask_5 and suggest improvements for simplification and consolidation."
    )
    cot_reflect_desc_1_1 = {
        'instruction': cot_reflect_instruction_1_1,
        'critic_instruction': critic_instruction_1_1,
        'input': [taskInfo] + loop_results['stage_0.subtask_5']['thinking'] + loop_results['stage_0.subtask_5']['answer'],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_0.subtask_5', 'answer of stage_0.subtask_5']
    }
    results_1_1, log_1_1 = await self.reflexion(
        subtask_id='stage_1.subtask_1',
        reflect_desc=cot_reflect_desc_1_1,
        n_repeat=self.max_round
    )
    logs.append(log_1_1)

    cot_reflect_instruction_1_2 = (
        "Sub-task 2: Evaluate each candidate sequence against strict criteria including regioselectivity, overall yield, reaction compatibility, and chemical feasibility, with special attention to the order of Friedel-Crafts alkylation and nitration, and the correct installation of the ethoxy group. "
        "Input: thinking and answer from stage_1.subtask_1"
    )
    critic_instruction_1_2 = (
        "Please review and provide limitations and evaluation of candidate sequences from stage_1.subtask_1, focusing on chemical feasibility and yield."
    )
    cot_reflect_desc_1_2 = {
        'instruction': cot_reflect_instruction_1_2,
        'critic_instruction': critic_instruction_1_2,
        'input': [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1']
    }
    results_1_2, log_1_2 = await self.reflexion(
        subtask_id='stage_1.subtask_2',
        reflect_desc=cot_reflect_desc_1_2,
        n_repeat=self.max_round
    )
    logs.append(log_1_2)

    cot_agent_instruction_1_3 = (
        "Sub-task 3: Select the best candidate sequence that leads to a high-yield synthesis of 2-(tert-butyl)-1-ethoxy-3-nitrobenzene, ensuring the sequence is chemically valid and consistent with all directing effects and reaction compatibility constraints. "
        "Input: thinking and answer from stage_1.subtask_2"
    )
    cot_agent_desc_1_3 = {
        'instruction': cot_agent_instruction_1_3,
        'input': [taskInfo, results_1_2['thinking'], results_1_2['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_1.subtask_2', 'answer of stage_1.subtask_2']
    }
    results_1_3, log_1_3 = await self.answer_generate(
        subtask_id='stage_1.subtask_3',
        cot_agent_desc=cot_agent_desc_1_3
    )
    logs.append(log_1_3)

    review_instruction_2_1 = (
        "Sub-task 1: Apply a systematic evaluation to the selected sequence from stage_1.subtask_3 to verify correctness and feasibility based on organic chemistry principles, including checks for ring activation/deactivation, directing effects, and functional group compatibility. "
        "Input: thinking and answer from stage_1.subtask_3"
    )
    review_desc_2_1 = {
        'instruction': review_instruction_2_1,
        'input': [taskInfo, results_1_3['thinking'], results_1_3['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_1.subtask_3', 'answer of stage_1.subtask_3']
    }
    results_2_1, log_2_1 = await self.review(
        subtask_id='stage_2.subtask_1',
        review_desc=review_desc_2_1
    )
    logs.append(log_2_1)

    cot_agent_instruction_2_2 = (
        "Sub-task 2: Produce a final validity indicator and detailed justification for the selected reaction sequence, explicitly referencing how the sequence avoids previous failure reasons such as incorrect step order, overuse of sulfonation, and incompatible substitutions. "
        "Input: thinking and answer from stage_2.subtask_1"
    )
    cot_agent_desc_2_2 = {
        'instruction': cot_agent_instruction_2_2,
        'input': [taskInfo, results_2_1['thinking'], results_2_1['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_2.subtask_1', 'answer of stage_2.subtask_1']
    }
    results_2_2, log_2_2 = await self.answer_generate(
        subtask_id='stage_2.subtask_2',
        cot_agent_desc=cot_agent_desc_2_2
    )
    logs.append(log_2_2)

    final_answer = await self.make_final_answer(results_2_2['thinking'], results_2_2['answer'])
    return final_answer, logs
