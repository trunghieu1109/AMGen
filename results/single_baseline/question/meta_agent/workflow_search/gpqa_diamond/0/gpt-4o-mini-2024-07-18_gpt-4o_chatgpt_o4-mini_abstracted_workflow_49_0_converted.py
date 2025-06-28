async def forward_0(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    # Stage 1: Analyze and Classify Elements
    cot_instruction = "Sub-task 1: Retrieve 3D molecular structures for triisopropyl borate, quinuclidine, benzo[1,2-c:3,4-c:5,6-c]trifuran-1,3,4,6,7,9-hexaone, and triphenyleno[1,2-c:5,6-c:9,10-c]trifuran-1,3,6,8,11,13-hexaone using a reliable database."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_instruction, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, retrieving 3D structures, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Subtask 1 answer: ", sub_tasks[-1])
    
    # Stage 2: Evaluate and prioritize elements
    cot_sc_instruction = "Sub-task 2: Analyze the 3D structures obtained in Subtask 1 using a point-group detection tool to identify symmetry elements for each molecule."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": cot_sc_instruction, "context": ["user query", "thinking of subtask 1", "answer of subtask 1"], "agent_collaboration": "SC_CoT"}
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    for i in range(N):
        thinking2, answer2 = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, analyzing symmetry elements, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers.append(answer2.content)
        thinkingmapping[answer2.content] = thinking2
        answermapping[answer2.content] = answer2
    answer2_content = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[answer2_content]
    answer2 = answermapping[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Subtask 2 answer: ", sub_tasks[-1])
    
    # Stage 3: Debate and validate symmetry classification
    debate_instruction = "Sub-task 3: Debate the symmetry classification of triisopropyl borate, with one agent arguing for C₃h symmetry and another against it."
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max = self.max_round
    all_thinking3 = [[] for _ in range(N_max)]
    all_answer3 = [[] for _ in range(N_max)]
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": debate_instruction, "context": ["user query", "thinking of subtask 2", "answer of subtask 2"], "agent_collaboration": "Debate"}
    for r in range(N_max):
        for i, agent in enumerate(debate_agents):
            input_infos = [taskInfo, thinking2, answer2]
            thinking3, answer3 = await agent(input_infos, debate_instruction, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating symmetry classification, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking3[r].append(thinking3)
            all_answer3[r].append(answer3)
    final_decision_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent([taskInfo] + all_thinking3[-1] + all_answer3[-1], "Sub-task 3: Make final decision on symmetry classification.", is_sub_task=True)
    agents.append(f"Final Decision agent, calculating final output, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Subtask 3 answer: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs