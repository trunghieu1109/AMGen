async def forward_160(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    logs = []
    loop_results = {
        'stage_0.subtask_1': {'thinking': [], 'answer': []},
        'stage_0.subtask_2': {'thinking': [], 'answer': []},
        'stage_0.subtask_3': {'thinking': [], 'answer': []},
        'stage_0.subtask_4': {'thinking': [], 'answer': []},
        'stage_0.subtask_5': {'thinking': [], 'answer': []}
    }

    for iteration in range(2):
        cot_instruction_1 = (
            "Sub-task 1: Extract and summarize all relevant given information from the query, including physical parameters, vacuum conditions, and observed phenomena. "
            "Ensure clarity on the definitions of λ1 and λ2 as presented in the problem to avoid ambiguity. "
            "Input content: taskInfo"
        )
        cot_agent_desc_1 = {
            "instruction": cot_instruction_1,
            "input": [taskInfo],
            "temperature": 0.0,
            "context": ["user query"]
        }
        results_1, log_1 = await self.cot(
            subtask_id="stage_0.subtask_1",
            cot_agent_desc=cot_agent_desc_1
        )
        loop_results['stage_0.subtask_1']['thinking'].append(results_1['thinking'])
        loop_results['stage_0.subtask_1']['answer'].append(results_1['answer'])
        logs.append(log_1)

        cot_instruction_2 = (
            "Sub-task 2: Analyze and clearly define the physical meaning of the initial mean free path λ1, specifically as the mean free path of gas molecules colliding with each other under ultra-high vacuum conditions. "
            "Emphasize that λ1 depends on gas density and gas-gas collision cross-section σ_gas-gas. Avoid conflating λ1 with electron scattering mean free path. "
            "Input content: results (thinking and answer) from stage_0.subtask_1 and all previous iterations of stage_0.subtask_4"
        )
        cot_agent_desc_2 = {
            "instruction": cot_instruction_2,
            "input": [taskInfo] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_1']['answer'] + loop_results['stage_0.subtask_4']['thinking'] + loop_results['stage_0.subtask_4']['answer'],
            "temperature": 0.0,
            "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1", "thinking of stage_0.subtask_4", "answer of stage_0.subtask_4"]
        }
        results_2, log_2 = await self.cot(
            subtask_id="stage_0.subtask_2",
            cot_agent_desc=cot_agent_desc_2
        )
        loop_results['stage_0.subtask_2']['thinking'].append(results_2['thinking'])
        loop_results['stage_0.subtask_2']['answer'].append(results_2['answer'])
        logs.append(log_2)

        cot_instruction_3 = (
            "Sub-task 3: Analyze and clearly define the mean free path λ2 as the mean free path of electrons scattering off residual gas molecules in the vacuum. "
            "Explicitly distinguish λ2 from λ1 by highlighting the different scattering cross-section σ_electron-gas and particle types involved. "
            "Discuss how electron beam initiation affects scattering dynamics without changing temperature. "
            "Input content: results (thinking and answer) from stage_0.subtask_2 and all previous iterations of stage_0.subtask_4"
        )
        cot_agent_desc_3 = {
            "instruction": cot_instruction_3,
            "input": [taskInfo] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'] + loop_results['stage_0.subtask_4']['thinking'] + loop_results['stage_0.subtask_4']['answer'],
            "temperature": 0.0,
            "context": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2", "thinking of stage_0.subtask_4", "answer of stage_0.subtask_4"]
        }
        results_3, log_3 = await self.cot(
            subtask_id="stage_0.subtask_3",
            cot_agent_desc=cot_agent_desc_3
        )
        loop_results['stage_0.subtask_3']['thinking'].append(results_3['thinking'])
        loop_results['stage_0.subtask_3']['answer'].append(results_3['answer'])
        logs.append(log_3)

        cot_instruction_4 = (
            "Sub-task 4: Perform a quantitative derivation of λ1 and λ2 using the kinetic theory formula λ = 1/(nσ), where n is the gas molecule density and σ is the relevant scattering cross-section. "
            "Use representative or literature values for σ_gas-gas and σ_electron-gas at the given vacuum pressure (<10^-9 Torr) and temperature. "
            "Calculate the ratio λ2/λ1 explicitly to determine if λ2 is greater than, equal to, or less than λ1, and whether any numeric factor like 1.22 is justified. "
            "Avoid arbitrary assumptions and clearly document all values and steps. "
            "Input content: results (thinking and answer) from stage_0.subtask_3, stage_0.subtask_2, stage_0.subtask_1 and all previous iterations of stage_0.subtask_4"
        )
        cot_agent_desc_4 = {
            "instruction": cot_instruction_4,
            "input": [taskInfo] + loop_results['stage_0.subtask_3']['thinking'] + loop_results['stage_0.subtask_3']['answer'] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_1']['answer'] + loop_results['stage_0.subtask_4']['thinking'] + loop_results['stage_0.subtask_4']['answer'],
            "temperature": 0.0,
            "context": ["user query", "thinking of stage_0.subtask_3", "answer of stage_0.subtask_3", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1", "thinking of stage_0.subtask_4", "answer of stage_0.subtask_4"]
        }
        results_4, log_4 = await self.structured_cot(
            subtask_id="stage_0.subtask_4",
            cot_agent_desc=cot_agent_desc_4
        )
        loop_results['stage_0.subtask_4']['thinking'].append(results_4['thinking'])
        loop_results['stage_0.subtask_4']['answer'].append(results_4['answer'])
        logs.append(log_4)

        cot_instruction_5 = (
            "Sub-task 5: Critically review and validate the assumptions and results from the quantitative derivation. "
            "Specifically, verify the physical definitions of λ1 and λ2, assess whether they are directly comparable, and consider secondary effects of electron beam initiation such as gas desorption, ionization, or changes in gas composition/density that might affect the mean free paths. "
            "This subtask aims to catch conceptual errors and prevent oversimplified conclusions as highlighted in the feedback. "
            "Input content: results (thinking and answer) from stage_0.subtask_4"
        )
        cot_agent_desc_5 = {
            "instruction": cot_instruction_5,
            "critic_instruction": "Please review and provide the limitations of provided solutions of stage_0.subtask_4.",
            "input": [taskInfo] + loop_results['stage_0.subtask_4']['thinking'] + loop_results['stage_0.subtask_4']['answer'],
            "temperature": 0.0,
            "context": ["user query", "thinking of stage_0.subtask_4", "answer of stage_0.subtask_4"]
        }
        results_5, log_5 = await self.reflexion(
            subtask_id="stage_0.subtask_5",
            reflect_desc=cot_agent_desc_5,
            n_repeat=self.max_round
        )
        loop_results['stage_0.subtask_5']['thinking'].append(results_5['thinking'])
        loop_results['stage_0.subtask_5']['answer'].append(results_5['answer'])
        logs.append(log_5)

    debate_instruction_1 = (
        "Sub-task 1: Evaluate the candidate answer choices against the refined and validated hypothesis about λ2 derived from previous subtasks. "
        "Select the best fitting conclusion based on quantitative and conceptual analysis, ensuring no unsupported assumptions or arbitrary numeric bounds are used. "
        "Input content: results (thinking and answer) from stage_0.subtask_5"
    )
    debate_desc_1 = {
        "instruction": debate_instruction_1,
        "final_decision_instruction": "Sub-task 1: Select the best candidate answer choice for the query based on previous analysis.",
        "input": [taskInfo] + loop_results['stage_0.subtask_5']['thinking'] + loop_results['stage_0.subtask_5']['answer'],
        "context": ["user query", "thinking of stage_0.subtask_5", "answer of stage_0.subtask_5"],
        "temperature": 0.5
    }
    results_final, log_final = await self.debate(
        subtask_id="stage_1.subtask_1",
        debate_desc=debate_desc_1,
        n_repeat=self.max_round
    )
    logs.append(log_final)

    final_answer = await self.make_final_answer(results_final['thinking'], results_final['answer'])
    return final_answer, logs
