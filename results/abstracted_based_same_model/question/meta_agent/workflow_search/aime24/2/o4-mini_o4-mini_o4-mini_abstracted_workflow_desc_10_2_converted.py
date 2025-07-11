async def forward_2(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction = "Sub-task 1: Label the eight vertices of the regular octagon cyclically as 0,1,…,7."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_instruction, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, labeling vertices, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction = "Sub-task 2: Define the eight rotations r_k(i) = (i + k) mod 8 for k = 0,…,7 in terms of vertex labels."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinking_map = {}
    answer_map = {}
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": cot_sc_instruction, "context": ["subtask_1"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking2, answer2 = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, defining rotations, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers.append(answer2.content)
        thinking_map[answer2.content] = thinking2
        answer_map[answer2.content] = answer2
    answer2_content = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinking_map[answer2_content]
    answer2 = answer_map[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_instruction3 = "Sub-task 3: Compute the total number of 2-colorings of 8 vertices: 2^8."
    cot_agent3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": cot_instruction3, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking3, answer3 = await cot_agent3([taskInfo], cot_instruction3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent3.id}, computing total colorings, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_sc_instruction4 = "Sub-task 4: Partition the 256 colorings by number k of blue vertices and count C(8,k) for k = 0,…,8."
    M = self.max_sc
    cot_agents4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(M)]
    possible_answers4 = []
    thinking_map4 = {}
    answer_map4 = {}
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": cot_sc_instruction4, "context": ["subtask_3"], "agent_collaboration": "SC_CoT"}
    for i in range(M):
        thinking4, answer4 = await cot_agents4[i]([taskInfo, thinking3, answer3], cot_sc_instruction4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents4[i].id}, partitioning colorings, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers4.append(answer4.content)
        thinking_map4[answer4.content] = thinking4
        answer_map4[answer4.content] = answer4
    answer4_content = Counter(possible_answers4).most_common(1)[0][0]
    thinking4 = thinking_map4[answer4_content]
    answer4 = answer_map4[answer4_content]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    cot_instruction5 = "Sub-task 5: For k=0,1,2,3 show that all C(8,k) colorings are valid by finding a nonzero rotation that avoids mapping any blue onto a blue and compute the sum of C(8,k)."
    cot_agent5 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {"subtask_id": "subtask_5", "instruction": cot_instruction5, "context": ["subtask_4"], "agent_collaboration": "CoT"}
    thinking5, answer5 = await cot_agent5([taskInfo, thinking4, answer4], cot_instruction5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent5.id}, counting valid for k<=3, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    cot_instruction61 = "Sub-task 6.1: List all 70 subsets B of size 4 from {0,…,7}."
    cot_agent61 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc61 = {"subtask_id": "subtask_6.1", "instruction": cot_instruction61, "context": ["subtask_4"], "agent_collaboration": "CoT"}
    thinking61, answer61 = await cot_agent61([taskInfo, thinking4, answer4], cot_instruction61, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent61.id}, listing subsets of size 4, thinking: {thinking61.content}; answer: {answer61.content}")
    sub_tasks.append(f"Sub-task 6.1 output: thinking - {thinking61.content}; answer - {answer61.content}")
    subtask_desc61['response'] = {"thinking": thinking61, "answer": answer61}
    logs.append(subtask_desc61)
    print("Step 6: ", sub_tasks[-1])

    cot_sc_instruction62 = "Sub-task 6.2: For each size-4 subset B, check if there exists k in {1,…,7} such that for every i in B, (i + k) mod 8 ∉ B."
    P = self.max_sc
    cot_agents62 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(P)]
    possible_answers62 = []
    thinking_map62 = {}
    answer_map62 = {}
    subtask_desc62 = {"subtask_id": "subtask_6.2", "instruction": cot_sc_instruction62, "context": ["subtask_2","subtask_6.1"], "agent_collaboration": "SC_CoT"}
    for i in range(P):
        thinking62, answer62 = await cot_agents62[i]([taskInfo, thinking2, answer2, thinking61, answer61], cot_sc_instruction62, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents62[i].id}, checking validity of size-4 subsets, thinking: {thinking62.content}; answer: {answer62.content}")
        possible_answers62.append(answer62.content)
        thinking_map62[answer62.content] = thinking62
        answer_map62[answer62.content] = answer62
    answer62_content = Counter(possible_answers62).most_common(1)[0][0]
    thinking62 = thinking_map62[answer62_content]
    answer62 = answer_map62[answer62_content]
    sub_tasks.append(f"Sub-task 6.2 output: thinking - {thinking62.content}; answer - {answer62.content}")
    subtask_desc62['response'] = {"thinking": thinking62, "answer": answer62}
    logs.append(subtask_desc62)
    print("Step 7: ", sub_tasks[-1])

    cot_instruction63 = "Sub-task 6.3: Count how many of the 70 subsets satisfy the validity condition from subtask_6.2."
    cot_agent63 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc63 = {"subtask_id": "subtask_6.3", "instruction": cot_instruction63, "context": ["subtask_6.2"], "agent_collaboration": "CoT"}
    thinking63, answer63 = await cot_agent63([taskInfo, thinking62, answer62], cot_instruction63, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent63.id}, counting valid size-4 subsets, thinking: {thinking63.content}; answer: {answer63.content}")
    sub_tasks.append(f"Sub-task 6.3 output: thinking - {thinking63.content}; answer - {answer63.content}")
    subtask_desc63['response'] = {"thinking": thinking63, "answer": answer63}
    logs.append(subtask_desc63)
    print("Step 8: ", sub_tasks[-1])

    cot_instruction7 = "Sub-task 7: Compute total valid colorings by summing valid counts for k=0–3 and the valid size-4 count."
    cot_agent7 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc7 = {"subtask_id": "subtask_7", "instruction": cot_instruction7, "context": ["subtask_5","subtask_6.3"], "agent_collaboration": "CoT"}
    thinking7, answer7 = await cot_agent7([taskInfo, thinking5, answer5, thinking63, answer63], cot_instruction7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent7.id}, summing valid colorings, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 9: ", sub_tasks[-1])

    cot_instruction8 = "Sub-task 8: Form probability P = (number of valid colorings) / 256."
    cot_agent8 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc8 = {"subtask_id": "subtask_8", "instruction": cot_instruction8, "context": ["subtask_3","subtask_7"], "agent_collaboration": "CoT"}
    thinking8, answer8 = await cot_agent8([taskInfo, thinking3, answer3, thinking7, answer7], cot_instruction8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent8.id}, forming probability, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 10: ", sub_tasks[-1])

    cot_instruction9 = "Sub-task 9: Reduce the fraction from subtask_8 to lowest terms m/n."
    cot_agent9 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc9 = {"subtask_id": "subtask_9", "instruction": cot_instruction9, "context": ["subtask_8"], "agent_collaboration": "CoT"}
    thinking9, answer9 = await cot_agent9([taskInfo, thinking8, answer8], cot_instruction9, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent9.id}, reducing fraction, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    subtask_desc9['response'] = {"thinking": thinking9, "answer": answer9}
    logs.append(subtask_desc9)
    print("Step 11: ", sub_tasks[-1])

    cot_instruction10 = "Sub-task 10: Calculate and return the integer m + n as the final answer."
    cot_agent10 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc10 = {"subtask_id": "subtask_10", "instruction": cot_instruction10, "context": ["subtask_9"], "agent_collaboration": "CoT"}
    thinking10, answer10 = await cot_agent10([taskInfo, thinking9, answer9], cot_instruction10, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent10.id}, computing m+n, thinking: {thinking10.content}; answer: {answer10.content}")
    sub_tasks.append(f"Sub-task 10 output: thinking - {thinking10.content}; answer - {answer10.content}")
    subtask_desc10['response'] = {"thinking": thinking10, "answer": answer10}
    logs.append(subtask_desc10)
    print("Step 12: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking10, answer10, sub_tasks, agents)
    return final_answer, logs