async def forward_150(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 1: Normalize the state vector (SC_CoT)
    N1 = self.max_sc
    cot_agents1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N1)]
    possible_thinkings1 = []
    possible_answers1 = []
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": "Sub-task 1: Normalize the given state vector ψ = (-1, 2, 1)^T by computing its norm explicitly, dividing each component by this norm, and verifying unit length.", "context": ["user query"], "agent_collaboration": "SC_CoT"}
    for i in range(N1):
        thinking_i, answer_i = await cot_agents1[i]([taskInfo], subtask_desc1["instruction"], is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents1[i].id}, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_thinkings1.append(thinking_i)
        possible_answers1.append(answer_i)
    final_agent1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    instr1 = "Sub-task 1: Synthesize and choose the most consistent and correct normalization result given all agent outputs."
    thinking1, answer1 = await final_agent1([taskInfo] + possible_thinkings1 + possible_answers1, instr1, is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Stage 2.1: Compute eigenvalues (SC_CoT)
    N2 = self.max_sc
    cot_agents2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_thinkings2 = []
    possible_answers2 = []
    subtask_desc2_1 = {"subtask_id": "subtask_2.1", "instruction": "Sub-task 2.1: Compute all eigenvalues of P by solving det(P - λI) = 0, show the characteristic polynomial and its roots, confirm that 0 is an eigenvalue and determine its degeneracy.", "context": ["user query"], "agent_collaboration": "SC_CoT"}
    for i in range(N2):
        thinking_i, answer_i = await cot_agents2[i]([taskInfo], subtask_desc2_1["instruction"], is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents2[i].id}, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_thinkings2.append(thinking_i)
        possible_answers2.append(answer_i)
    final_agent2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    instr2 = "Sub-task 2.1: Synthesize and choose the most consistent eigenvalue calculation result."
    thinking2_1, answer2_1 = await final_agent2([taskInfo] + possible_thinkings2 + possible_answers2, instr2, is_sub_task=True)
    sub_tasks.append(f"Sub-task 2.1 output: thinking - {thinking2_1.content}; answer - {answer2_1.content}")
    subtask_desc2_1['response'] = {"thinking": thinking2_1, "answer": answer2_1}
    logs.append(subtask_desc2_1)
    print("Step 2.1: ", sub_tasks[-1])

    # Stage 2.2: Find eigenvector for eigenvalue 0 (SC_CoT)
    N3 = self.max_sc
    cot_agents3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N3)]
    possible_thinkings3 = []
    possible_answers3 = []
    subtask_desc2_2 = {"subtask_id": "subtask_2.2", "instruction": "Sub-task 2.2: Solve (P - 0·I)v = 0, find and normalize the eigenvector v ∝ [1,0,-1]/√2, and verify normalization.", "context": ["user query", "thinking of subtask 2.1", "answer of subtask 2.1"], "agent_collaboration": "SC_CoT"}
    for i in range(N3):
        thinking_i, answer_i = await cot_agents3[i]([taskInfo, thinking2_1, answer2_1], subtask_desc2_2["instruction"], is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents3[i].id}, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_thinkings3.append(thinking_i)
        possible_answers3.append(answer_i)
    final_agent3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    instr3 = "Sub-task 2.2: Synthesize and choose the most consistent eigenvector normalization result."
    thinking2_2, answer2_2 = await final_agent3([taskInfo, thinking2_1, answer2_1] + possible_thinkings3 + possible_answers3, instr3, is_sub_task=True)
    sub_tasks.append(f"Sub-task 2.2 output: thinking - {thinking2_2.content}; answer - {answer2_2.content}")
    subtask_desc2_2['response'] = {"thinking": thinking2_2, "answer": answer2_2}
    logs.append(subtask_desc2_2)
    print("Step 2.2: ", sub_tasks[-1])

    # Stage 3: Build projector P0 (SC_CoT)
    N4 = self.max_sc
    cot_agents4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N4)]
    possible_thinkings4 = []
    possible_answers4 = []
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": "Sub-task 3: Construct the projector P0 = v·v^T onto the zero-eigenspace using the normalized eigenvector v, and display all nine matrix entries.", "context": ["user query", "thinking of subtask 2.2", "answer of subtask 2.2"], "agent_collaboration": "SC_CoT"}
    for i in range(N4):
        thinking_i, answer_i = await cot_agents4[i]([taskInfo, thinking2_2, answer2_2], subtask_desc3["instruction"], is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents4[i].id}, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_thinkings4.append(thinking_i)
        possible_answers4.append(answer_i)
    final_agent4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    instr4 = "Sub-task 3: Synthesize and choose the most consistent projector construction result."
    thinking3, answer3 = await final_agent4([taskInfo, thinking2_2, answer2_2] + possible_thinkings4 + possible_answers4, instr4, is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    # Stage 4: Compute probability (SC_CoT)
    N5 = self.max_sc
    cot_agents5 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N5)]
    possible_thinkings5 = []
    possible_answers5 = []
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": "Sub-task 4: Apply P0 to the normalized ψ, compute the intermediate vector and ⟨ψ|P0|ψ⟩, then square the magnitude to get the probability, showing all steps.", "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 3", "answer of subtask 3"], "agent_collaboration": "SC_CoT"}
    for i in range(N5):
        thinking_i, answer_i = await cot_agents5[i]([taskInfo, thinking1, answer1, thinking3, answer3], subtask_desc4["instruction"], is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents5[i].id}, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_thinkings5.append(thinking_i)
        possible_answers5.append(answer_i)
    final_agent5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    instr5 = "Sub-task 4: Synthesize and choose the most consistent probability computation result."
    thinking4, answer4 = await final_agent5([taskInfo, thinking1, answer1, thinking3, answer3] + possible_thinkings5 + possible_answers5, instr5, is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    # Stage 5: Compare with choices and select (Debate)
    debate_instr = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    base_instruction5 = "Sub-task 5: Compare the calculated probability with the provided choices {1/3, 2/3, √(2/3), 1} and select the correct one, citing the numeric match."
    debate_instruction5 = base_instruction5 + " " + debate_instr
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking5 = [[] for _ in range(self.max_round)]
    all_answer5 = [[] for _ in range(self.max_round)]
    subtask_desc5 = {"subtask_id": "subtask_5", "instruction": debate_instruction5, "context": ["user query", "thinking of subtask 4", "answer of subtask 4"], "agent_collaboration": "Debate"}
    for r in range(self.max_round):
        for agent in debate_agents:
            if r == 0:
                thinking_r, answer_r = await agent([taskInfo, thinking4, answer4], debate_instruction5, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking_r, answer_r = await agent(inputs, debate_instruction5, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking_r.content}; answer: {answer_r.content}")
            all_thinking5[r].append(thinking_r)
            all_answer5[r].append(answer_r)
    final_decider = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr5 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking5, answer5 = await final_decider([taskInfo, thinking4, answer4] + all_thinking5[-1] + all_answer5[-1], base_instruction5 + " " + final_instr5, is_sub_task=True)
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs