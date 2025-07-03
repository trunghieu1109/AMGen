async def forward_196(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Subtask 1: Analyze the IR spectrum of Compound X, identifying and validating key functional groups such as carboxylic acid, aromatic rings, and others, "
        "based on the given IR data and task context. Confirm that identified groups align with typical IR peak assignments."
    )
    cot_agent_desc_1 = {
        'instruction': cot_instruction_1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc_1
    )
    agents.append(f"CoT agent {results1['cot_agent'].id}, analyzing IR spectrum, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_sc_instruction_2 = (
        "Subtask 2: Analyze the 1H NMR spectrum of Compound X, extracting key proton environments, splitting patterns, and integration values. "
        "Cross-check integration and splitting patterns to confirm substitution patterns and structural features, considering the task context."
    )
    cot_sc_desc_2 = {
        'instruction': cot_sc_instruction_2,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_sc_desc=cot_sc_desc_2,
        n_repeat=self.max_sc
    )
    for idx in range(self.max_sc):
        agents.append(f"CoT-SC agent {results2['cot_agent'][idx].id}, analyzing NMR spectrum, thinking: {results2['list_thinking'][idx]}; answer: {results2['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    cot_instruction_3 = (
        "Subtask 3: Integrate the IR and NMR data to deduce the likely structure and functional groups present in Compound X before reaction. "
        "Explicitly reconcile any conflicting data between IR and NMR outputs from Subtasks 1 and 2 before proceeding."
    )
    cot_sc_desc_3 = {
        'instruction': cot_instruction_3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_sc_desc=cot_sc_desc_3,
        n_repeat=self.max_sc
    )
    for idx in range(self.max_sc):
        agents.append(f"CoT-SC agent {results3['cot_agent'][idx].id}, integrating IR and NMR data, thinking: {results3['list_thinking'][idx]}; answer: {results3['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    debate_instruction_4 = (
        "Subtask 4: Analyze the reaction conditions involving red phosphorus and HI to determine the chemical transformation Compound X undergoes. "
        "Incorporate explicit chemical knowledge that red phosphorus and HI induce decarboxylative reduction of aromatic carboxylic acids to hydrocarbons (e.g., Hunsdiecker or Bouveault–Blanc reaction). "
        "Consider complete reduction scenarios and cite relevant literature precedents to guide mechanistic interpretation. "
        "Self-reflect on the plausibility of the mechanism and expected product before finalizing."
    )
    debate_desc_4 = {
        'instruction': debate_instruction_4,
        'context': ["user query"],
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc_4,
        final_decision_desc={
            'instruction': "Subtask 4: Make final decision on the type of chemical transformation Compound X undergoes, explicitly considering decarboxylative reduction.",
            'output': ["thinking", "answer"],
            'temperature': 0.0
        },
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results4['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, analyzing reaction conditions, thinking: {results4['list_thinking'][round][idx].content}; answer: {results4['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, deciding reaction type, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])

    cot_reflect_instruction_5 = (
        "Subtask 5: Predict the final product structure of Compound X after reaction with red phosphorus and HI, based on the deduced initial structure and reaction type from Subtasks 3 and 4. "
        "Explicitly compare the predicted product with the given multiple-choice options, discuss any discrepancies, and justify the final selection or indicate if no option matches."
    )
    critic_instruction_5 = (
        "Please review the predicted final product and its comparison with the provided multiple-choice options. Identify any discrepancies or unsupported predictions and provide feedback."
    )
    cot_reflect_desc_5 = {
        'instruction': cot_reflect_instruction_5,
        'input': [taskInfo, results3['thinking'], results3['answer'], results4['thinking'], results4['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 4", "answer of subtask 4"]
    }
    critic_desc_5 = {
        'instruction': critic_instruction_5,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results5 = await self.reflexion(
        subtask_id="subtask_5",
        cot_reflect_desc=cot_reflect_desc_5,
        critic_desc=critic_desc_5,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results5['cot_agent'].id}, predicting final product and comparing with options, thinking: {results5['list_thinking'][0].content}; answer: {results5['list_answer'][0].content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {results5['critic_agent'].id}, providing feedback, thinking: {results5['list_feedback'][i].content}; answer: {results5['list_correct'][i].content}")
        agents.append(f"Reflexion CoT agent {results5['cot_agent'].id}, refining prediction and justification, thinking: {results5['list_thinking'][i + 1].content}; answer: {results5['list_answer'][i + 1].content}")
    sub_tasks.append(f"Subtask 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])

    cot_sc_instruction_6 = (
        "Subtask 6: Based on the predicted final product and comparison with multiple-choice options, provide at least three independent justifications for mapping the predicted product to one of the options. "
        "Perform a majority vote to select the final answer, explicitly avoiding selecting the starting material or unsupported choices."
    )
    cot_sc_desc_6 = {
        'instruction': cot_sc_instruction_6,
        'input': [taskInfo, results5['thinking'], results5['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 5", "answer of subtask 5"]
    }
    results6 = await self.sc_cot(
        subtask_id="subtask_6",
        cot_sc_desc=cot_sc_desc_6,
        n_repeat=self.max_sc
    )
    for idx in range(self.max_sc):
        agents.append(f"CoT-SC agent {results6['cot_agent'][idx].id}, providing justification {idx+1}, thinking: {results6['list_thinking'][idx]}; answer: {results6['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 6 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
    logs.append(results6['subtask_desc'])

    final_answer = await self.make_final_answer(results6['thinking'], results6['answer'], sub_tasks, agents)
    return final_answer, logs
