async def forward_0(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    """
    [Stage 1: Molecular Structure Verification and Symmetry Identification]
    [Objective]
    - Collect and verify the molecular structures and chemical formulas of each given molecule (quinuclidine, triisopropyl borate, benzo[1,2-c:3,4-c':5,6-c'']trifuran-1,3,4,6,7,9-hexaone, triphenyleno[1,2-c:5,6-c':9,10-c'']trifuran-1,3,6,8,11,13-hexaone) to ensure accurate identification of their symmetry elements.
    - Determine the point group symmetry of each molecule based on their verified molecular structures and chemical formulas, focusing on identifying symmetry elements such as rotation axes, mirror planes, and inversion centers.
    [Agent Collaborations]
    - Use Chain-of-Thought (CoT) for initial verification of molecular structures.
    - Use Self-Consistency Chain-of-Thought (SC-CoT) for determining point group symmetries to ensure robustness.
    """

    cot_instruction_1 = "Sub-task 1: Collect and verify the molecular structures and chemical formulas of each molecule (quinuclidine, triisopropyl borate, benzo[1,2-c:3,4-c':5,6-c'']trifuran-1,3,4,6,7,9-hexaone, triphenyleno[1,2-c:5,6-c':9,10-c'']trifuran-1,3,6,8,11,13-hexaone) to ensure accurate identification of their symmetry elements."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, verifying molecular structures, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Subtask 1 answer: ", sub_tasks[-1])

    cot_sc_instruction_2 = "Sub-task 2: Based on verified molecular structures, determine the point group symmetry of each molecule, focusing on identifying symmetry elements such as rotation axes, mirror planes, and inversion centers."
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, determining point group symmetries, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2_content]
    answer2 = answermapping_2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)
    print("Subtask 2 answer: ", sub_tasks[-1])

    """
    [Stage 2: C3 Symmetry Analysis and Final Selection]
    [Objective]
    - Analyze the identified point group symmetries from subtask_2 to check specifically for the presence of C3 symmetry elements (a threefold rotational axis) in each molecule.
    - Compare the results of the C3 symmetry analysis for all molecules and select which molecule(s) exhibit C3h symmetry, thereby answering the original question.
    [Agent Collaborations]
    - Use Reflexion pattern to iteratively refine the analysis and selection based on feedback.
    """

    cot_reflect_instruction_3 = "Sub-task 3: Analyze the point group symmetries from Sub-task 2 to identify molecules exhibiting C3 symmetry elements and select which molecule(s) have C3h symmetry."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs_3 = [taskInfo, thinking2, answer2]
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, analyzing C3 symmetry and selecting molecules, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3],
                                               "Please review the C3 symmetry analysis and molecule selection for correctness and completeness.",
                                               i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining C3 symmetry analysis, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_reflect_instruction_3,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Reflexion",
        "response": {
            "thinking": thinking3,
            "answer": answer3
        }
    }
    logs.append(subtask_desc3)
    print("Subtask 3 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs
