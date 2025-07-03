async def forward_185(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction1 = (
        "Subtask 1: Analyze the 3D stereochemical arrangement of (1S,4R)-2-vinyl-2-azabicyclo[2.2.1]hept-5-ene, "
        "explicitly mapping which substituents are endo or exo, how the 1S,4R stereocenters fix the vinyl group orientation, "
        "describe the bicyclic framework stereochemistry, and relate these features to the multiple-choice product options."
    )
    results1_cot = await self.cot(
        subtask_id="subtask_1_cot",
        cot_instruction=cot_instruction1,
        input_list=[taskInfo],
        output_fields=["thinking", "answer"],
        temperature=0.0,
        context=["user query"]
    )

    cot_reflect_instruction1 = (
        "Subtask 1 Reflexion: Review and validate the stereochemical analysis of the starting material, "
        "checking for correct assignment of endo/exo orientations and consistency with 1S,4R stereocenters and product options."
    )
    critic_instruction1 = "Please provide feedback on any stereochemical misassignments or inconsistencies in the analysis."
    cot_reflect_desc1 = {
        'instruction': cot_reflect_instruction1,
        'input': [taskInfo, results1_cot['thinking'], results1_cot['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", results1_cot['thinking'], results1_cot['answer']]
    }
    critic_desc1 = {
        'instruction': critic_instruction1,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results1_reflex = await self.reflexion(
        subtask_id="subtask_1_reflex",
        cot_reflect_desc=cot_reflect_desc1,
        critic_desc=critic_desc1,
        n_repeat=self.max_round
    )

    agents.append(f"CoT agent {results1_cot['cot_agent'].id}, analyzing stereochemistry, thinking: {results1_cot['thinking'].content}; answer: {results1_cot['answer'].content}")
    for i in range(self.max_round):
        agents.append(f"Reflexion CoT agent {results1_reflex['cot_agent'].id}, refining stereochemical analysis, thinking: {results1_reflex['list_thinking'][i].content}; answer: {results1_reflex['list_answer'][i].content}")
        agents.append(f"Critic agent {results1_reflex['critic_agent'].id}, feedback, thinking: {results1_reflex['list_feedback'][i].content}; answer: {results1_reflex['list_correct'][i].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1_reflex['thinking'].content}; answer - {results1_reflex['answer'].content}")
    logs.append(results1_reflex['subtask_desc'])

    debate_instruction2 = (
        "Subtask 2: Two agents independently propose detailed transition-state models for the Cope rearrangement of the bicyclic azabicyclo compound: "
        "one assuming the vinyl group is endo, the other exo. Each must specify which C–C bonds break and form, how the nitrogen influences the mechanism, "
        "and the stereochemical consequences on the product. Then, agents critique each other's models focusing on stereochemical implications."
    )
    final_decision_instruction2 = "Subtask 2: Make a final decision on the most plausible transition-state model and mechanism based on the debate."
    debate_desc2 = {
        "instruction": debate_instruction2,
        "context": ["user query", results1_reflex['thinking'], results1_reflex['answer']],
        "input": [taskInfo, results1_reflex['thinking'], results1_reflex['answer']],
        "output": ["thinking", "answer"],
        "temperature": 0.5
    }
    final_decision_desc2 = {
        "instruction": final_decision_instruction2,
        "output": ["thinking", "answer"],
        "temperature": 0.0
    }
    results2 = await self.debate(
        subtask_id="subtask_2",
        debate_desc=debate_desc2,
        final_decision_desc=final_decision_desc2,
        n_repeat=self.max_round
    )

    for round in range(self.max_round):
        for idx, agent in enumerate(results2['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, proposing and critiquing transition-state models, thinking: {results2['list_thinking'][round][idx].content}; answer: {results2['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, selecting most plausible mechanism, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    cot_sc_instruction3 = (
        "Subtask 3: Independently predict the Cope rearrangement product structure based on the detailed mechanism and stereochemical analysis from Subtask 2. "
        "Three agents perform self-consistent Chain-of-Thought reasoning to justify their product choice, explicitly linking stereochemistry and mechanism. "
        "Use majority voting to select the final predicted product."
    )
    results3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_sc_instruction=cot_sc_instruction3,
        input_list=[taskInfo, results2['thinking'], results2['answer']],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user query", results2['thinking'], results2['answer']],
        n_repeat=3
    )

    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    for idx, cot_agent in enumerate(results3['cot_agent']):
        agents.append(f"SC-CoT agent {cot_agent.id}, independent product prediction #{idx+1}, thinking: {results3['list_thinking'][idx]}; answer: {results3['list_answer'][idx]}")
    logs.append(results3['subtask_desc'])

    debate_instruction4 = (
        "Subtask 4: Debate among agents on the four multiple-choice product options (A, B, C, D), focusing on stereochemical and mechanistic plausibility based on Subtask 3 output. "
        "Each agent argues for or against options, highlighting stereochemical consistency or contradictions."
    )
    final_decision_instruction4 = (
        "Subtask 4: Following the debate, each agent independently votes for the correct product option with justification using self-consistency CoT. "
        "Aggregate votes to finalize the answer."
    )
    debate_desc4 = {
        "instruction": debate_instruction4,
        "context": ["user query", results3['thinking'], results3['answer']],
        "input": [taskInfo, results3['thinking'], results3['answer']],
        "output": ["thinking", "answer"],
        "temperature": 0.5
    }
    final_decision_desc4 = {
        "instruction": final_decision_instruction4,
        "output": ["thinking", "answer"],
        "temperature": 0.5
    }
    results4_debate = await self.debate(
        subtask_id="subtask_4_debate",
        debate_desc=debate_desc4,
        final_decision_desc=final_decision_desc4,
        n_repeat=self.max_round
    )

    for round in range(self.max_round):
        for idx, agent in enumerate(results4_debate['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, debating product options, thinking: {results4_debate['list_thinking'][round][idx].content}; answer: {results4_debate['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, aggregating votes and finalizing product choice, thinking: {results4_debate['thinking'].content}; answer: {results4_debate['answer'].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4_debate['thinking'].content}; answer - {results4_debate['answer'].content}")
    logs.append(results4_debate['subtask_desc'])

    final_answer = await self.make_final_answer(results4_debate['thinking'], results4_debate['answer'].content.strip(), sub_tasks, agents)
    return final_answer, logs
