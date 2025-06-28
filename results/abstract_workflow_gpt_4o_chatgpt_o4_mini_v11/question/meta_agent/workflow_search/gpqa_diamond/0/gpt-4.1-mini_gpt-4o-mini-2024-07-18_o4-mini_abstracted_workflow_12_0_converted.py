async def forward_0(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    """
    [Stage 1: Collect and Analyze Molecular Geometry]
    [Objective] 
    - Collect detailed 3D structural information and molecular geometry data for each candidate molecule.
    - This includes bond lengths, bond angles, atomic coordinates, and conformational details.
    [Agent Collaborations]
    - Use Chain-of-Thought (CoT) to systematically gather and analyze structural data for each molecule.
    """
    
    cot_instruction_1 = (
        "Subtask 1: Collect detailed 3D structural information and molecular geometry data for each candidate molecule "
        "(quinuclidine, triisopropyl borate, benzo[1,2-c:3,4-c':5,6-c'']trifuran-1,3,4,6,7,9-hexaone, "
        "triphenyleno[1,2-c:5,6-c':9,10-c'']trifuran-1,3,6,8,11,13-hexaone). Include bond lengths, bond angles, atomic coordinates, and conformational details if available. "
        "Use context from the user query to focus on relevant structural features.")
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, collecting 3D structural data, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Subtask 1 answer: ", sub_tasks[-1])
    
    """
    [Stage 2: Identify and Verify Symmetry Elements]
    [Objective] 
    - Analyze the collected 3D structural data to identify all symmetry elements present in each molecule.
    - Verify the presence of a true C3 symmetry axis by testing 120° rotational symmetry on the 3D data.
    [Agent Collaborations]
    - Use Self-Consistency Chain-of-Thought (SC-CoT) to generate multiple independent symmetry analyses and confirm results by majority voting.
    """
    
    cot_sc_instruction_2 = (
        "Subtask 2: Analyze the 3D structural data from Subtask 1 to identify all symmetry elements present in each molecule, "
        "focusing on detecting C3 rotational axes and other relevant symmetry operations (mirror planes, inversion centers). "
        "Then verify the presence of a true C3 symmetry axis by testing 120° rotational symmetry on the 3D structure data, confirming that rotation maps the molecule onto itself within acceptable tolerance. "
        "Consider all substituents and stereochemical features."
    )
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
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, analyzing symmetry elements, thinking: {thinking2.content}; answer: {answer2.content}")
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
    [Stage 3: Integrate Results and Select Molecule with C3h Symmetry]
    [Objective] 
    - Integrate and compare symmetry analysis results to determine which molecule(s) possess genuine C3 symmetry.
    - Select exactly one molecule from the given choices that exhibits C3 symmetry.
    [Agent Collaborations]
    - Use Debate among multiple agents to synthesize perspectives and reach consensus on the final selection.
    """
    
    debate_instruction_3 = (
        "Subtask 3: Based on the structured summary from Subtask 2, integrate and compare the symmetry analysis results for all molecules. "
        "Determine which molecule(s) possess genuine C3 symmetry and select exactly one molecule from the given choices that exhibits C3 symmetry. "
        "Return the corresponding letter choice (A, B, C, or D) as a single uppercase character without any additional text or explanation."
    )
    debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3 = self.max_round
    all_thinking3 = [[] for _ in range(N_max_3)]
    all_answer3 = [[] for _ in range(N_max_3)]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": debate_instruction_3,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3):
        for i, agent in enumerate(debate_agents_3):
            if r == 0:
                thinking3, answer3 = await agent([taskInfo, thinking2, answer2], debate_instruction_3, r, is_sub_task=True)
            else:
                input_infos_3 = [taskInfo, thinking2, answer2] + all_thinking3[r-1] + all_answer3[r-1]
                thinking3, answer3 = await agent(input_infos_3, debate_instruction_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, selecting molecule with C3 symmetry, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking3[r].append(thinking3)
            all_answer3[r].append(answer3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo] + all_thinking3[-1] + all_answer3[-1], "Subtask 3: Make a final decision on the molecule with C3 symmetry.", is_sub_task=True)
    agents.append(f"Final Decision agent on molecule selection, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    print("Subtask 3 answer: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs
