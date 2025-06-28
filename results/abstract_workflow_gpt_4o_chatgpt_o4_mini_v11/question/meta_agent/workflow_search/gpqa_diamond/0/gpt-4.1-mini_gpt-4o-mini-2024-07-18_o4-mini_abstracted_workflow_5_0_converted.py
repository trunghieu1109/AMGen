
async def forward_0_0(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    """
    [Stage 1: Collect Molecular Structural Information]
    [Objective] 
    - Collect detailed molecular structural information and known symmetry elements for each given molecule: quinuclidine, triisopropyl borate, benzo[1,2-c:3,4-c':5,6-c'']trifuran-1,3,4,6,7,9-hexaone, and triphenyleno[1,2-c:5,6-c':9,10-c'']trifuran-1,3,6,8,11,13-hexaone.
    - Retrieve 3D geometries, point group assignments if available, and relevant literature or database references to establish a baseline understanding of their symmetry properties.
    [Agent Collaborations]
    - Chain-of-Thought (CoT)
    """
    cot_instruction = "Sub-task 1: Collect detailed molecular structural information and known symmetry elements for each molecule: quinuclidine, triisopropyl borate, benzo[1,2-c:3,4-c':5,6-c'']trifuran-1,3,4,6,7,9-hexaone, and triphenyleno[1,2-c:5,6-c':9,10-c'']trifuran-1,3,6,8,11,13-hexaone. Include 3D geometries, point group assignments if available, and relevant references."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, collecting molecular structural info, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Subtask 1 answer: ", sub_tasks[-1])
    
    """
    [Stage 2: Assign Point Group Symmetry]
    [Objective] 
    - Using the structural data from subtask_1, identify and assign the point group symmetry for each molecule.
    - Focus on detecting the presence or absence of the C3h point group by analyzing symmetry elements such as rotation axes, mirror planes, and improper axes.
    [Agent Collaborations]
    - Self-Consistency Chain-of-Thought (SC_CoT)
    """
    cot_sc_instruction = "Sub-task 2: Based on the structural data from Sub-task 1, identify and assign the point group symmetry for each molecule, focusing on detecting C3h symmetry by analyzing rotation axes, mirror planes, and improper axes."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2, answer2 = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, assigning point group symmetry, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers.append(answer2.content)
        thinkingmapping[answer2.content] = thinking2
        answermapping[answer2.content] = answer2
    most_common_answer = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[most_common_answer]
    answer2 = answermapping[most_common_answer]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)
    print("Subtask 2 answer: ", sub_tasks[-1])
    
    """
    [Stage 3: Verify C3h Symmetry Presence]
    [Objective] 
    - Examine the assigned point groups from subtask_2 to verify specifically whether each molecule possesses the C3h symmetry.
    - Confirm presence of a C3 principal rotation axis combined with a horizontal mirror plane (σh), and ensure no conflicting symmetry elements.
    [Agent Collaborations]
    - Reflexion
    """
    cot_reflect_instruction = "Sub-task 3: Verify for each molecule if it possesses C3h symmetry by confirming the presence of a C3 principal rotation axis and a horizontal mirror plane (σh), and absence of conflicting symmetry elements."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_reflect_instruction,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Reflexion"
    }
    thinking3, answer3 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, verifying C3h symmetry presence, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking3, answer3], "Please review the C3h symmetry verification and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining C3h verification, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    print("Subtask 3 answer: ", sub_tasks[-1])
    
    """
    [Stage 4: Select Molecule with C3h Symmetry]
    [Objective] 
    - Compare and contrast the symmetry verification results from subtask_3 across all molecules.
    - Determine which molecule(s) exhibit C3h symmetry.
    - Provide the final answer as a single letter choice (A, B, C, or D) without additional justification.
    [Agent Collaborations]
    - Debate
    """
    debate_instruction_4 = "Sub-task 4: Based on the verification results, select the molecule that exhibits C3h symmetry from the given choices: A) quinuclidine, B) triisopropyl borate, C) benzo[1,2-c:3,4-c':5,6-c'']trifuran-1,3,4,6,7,9-hexaone, D) triphenyleno[1,2-c:5,6-c':9,10-c'']trifuran-1,3,6,8,11,13-hexaone. Output only the letter choice without justification."
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max_4)]
    all_answer4 = [[] for _ in range(N_max_4)]
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": debate_instruction_4,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            input_infos_4 = [taskInfo, thinking3, answer3]
            thinking4, answer4 = await agent(input_infos_4, debate_instruction_4, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, selecting molecule with C3h symmetry, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1], "Sub-task 4: Make final decision on molecule selection.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting molecule, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    print("Subtask 4 answer: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs
                    