async def forward_8(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction_1 = (
        "Sub-task 1: Precisely define the game states and rules of transitions. "
        "Represent each state by the number of tokens remaining n, 0 ≤ n ≤ 2024. "
        "Allowed moves: remove 1 or 4 tokens. Winning condition: player removing last token wins. "
        "Assumptions: perfect play, no randomness. Do not solve or classify states here."
    )
    cot_sc_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(self.max_sc)]
    possible_answers_1 = []
    possible_thinkings_1 = []
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking1, answer1 = await cot_sc_agents_1[i]([taskInfo], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1[i].id}, defining game states and rules, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_answers_1.append(answer1)
        possible_thinkings_1.append(thinking1)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1([taskInfo] + possible_thinkings_1, "Sub-task 1: Synthesize and finalize game state definitions and rules.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = (
        "Sub-task 2: Formulate the theoretical framework of winning and losing positions. "
        "Define losing position as one where current player has no move to losing position (all moves lead to winning positions). "
        "Winning position: current player can move to at least one losing position. Recursive classification with base case dp[0] = losing. "
        "Do not compute actual positions here; focus on conceptual definitions and recursion logic."
    )
    cot_sc_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(self.max_sc)]
    possible_answers_2 = []
    possible_thinkings_2 = []
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking2, answer2 = await cot_sc_agents_2[i]([taskInfo, thinking1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_2[i].id}, formulating winning/losing framework, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2)
        possible_thinkings_2.append(thinking2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo] + possible_thinkings_2, "Sub-task 2: Synthesize and finalize winning/losing position framework.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    debate_instr_3 = (
        "Sub-task 3: Develop a precise algorithm to classify all positions n from 0 to 2024 as winning or losing using backward induction or dynamic programming. "
        "Specify recurrence: dp[n] = False (losing) iff all dp[n - move] are True (winning), moves in {1,4}, n - move ≥ 0; else dp[n] = True (winning). "
        "Define data structures for dp storage. Avoid actual computation here; focus on algorithm design and correctness. "
        "Given solutions from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_max_3 = self.max_round
    all_thinking_3 = [[] for _ in range(N_max_3)]
    all_answer_3 = [[] for _ in range(N_max_3)]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": debate_instr_3,
        "context": ["user query", thinking2.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3):
        for i, agent in enumerate(debate_agents_3):
            if r == 0:
                thinking3, answer3 = await agent([taskInfo, thinking2], debate_instr_3, r, is_sub_task=True)
            else:
                input_infos_3 = [taskInfo, thinking2] + all_thinking_3[r-1]
                thinking3, answer3 = await agent(input_infos_3, debate_instr_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, algorithm design, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking_3[r].append(thinking3)
            all_answer_3[r].append(answer3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo] + all_thinking_3[-1], "Sub-task 3: Synthesize and finalize algorithm design.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_instruction_4 = (
        "Sub-task 4: Implement the classification algorithm from Sub-task 3 to compute dp array for n=0 to 2024. "
        "Store results in boolean array dp. Emphasize correctness and efficiency. Avoid manual computations."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", thinking3.content],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, implementing dp classification, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    cot_instruction_5 = (
        "Sub-task 5: Perform a sanity check by enumerating dp values for n=0 to 15. "
        "Present results clearly to enable empirical observation of patterns. "
        "Avoid assumptions or conclusions; focus on accurate data presentation."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_instruction_5,
        "context": ["user query", thinking4.content],
        "agent_collaboration": "CoT"
    }
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking4], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, enumerating dp values 0 to 15, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    cot_instruction_6a = (
        "Sub-task 6a: Enumerate all losing positions n (1 ≤ n ≤ 2024) by iterating over dp array from Sub-task 4. "
        "Extract all indices where dp[n] == False. Prepare a complete, data-driven list without assumptions or modular arithmetic."
    )
    cot_agent_6a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6a = {
        "subtask_id": "subtask_6a",
        "instruction": cot_instruction_6a,
        "context": ["user query", thinking4.content],
        "agent_collaboration": "CoT"
    }
    thinking6a, answer6a = await cot_agent_6a([taskInfo, thinking4], cot_instruction_6a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6a.id}, enumerating losing positions, thinking: {thinking6a.content}; answer: {answer6a.content}")
    sub_tasks.append(f"Sub-task 6a output: thinking - {thinking6a.content}; answer - {answer6a.content}")
    subtask_desc6a['response'] = {"thinking": thinking6a, "answer": answer6a}
    logs.append(subtask_desc6a)
    print("Step 6a: ", sub_tasks[-1])

    debate_instr_6b = (
        "Sub-task 6b: Optionally analyze enumerated losing positions from Sub-task 6a to detect patterns. "
        "Rigorously verify any hypothesized patterns against enumerated data with multiple examples. "
        "If confirmed, document with explicit evidence; else reject premature generalizations. "
        "Do not influence counting or classification correctness. "
        "Given solutions from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_6b = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_max_6b = self.max_round
    all_thinking_6b = [[] for _ in range(N_max_6b)]
    all_answer_6b = [[] for _ in range(N_max_6b)]
    subtask_desc6b = {
        "subtask_id": "subtask_6b",
        "instruction": debate_instr_6b,
        "context": ["user query", thinking5.content, thinking6a.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_6b):
        for i, agent in enumerate(debate_agents_6b):
            if r == 0:
                thinking6b, answer6b = await agent([taskInfo, thinking5, thinking6a], debate_instr_6b, r, is_sub_task=True)
            else:
                input_infos_6b = [taskInfo, thinking5, thinking6a] + all_thinking_6b[r-1]
                thinking6b, answer6b = await agent(input_infos_6b, debate_instr_6b, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing losing position patterns, thinking: {thinking6b.content}; answer: {answer6b.content}")
            all_thinking_6b[r].append(thinking6b)
            all_answer_6b[r].append(answer6b)
    final_decision_agent_6b = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6b, answer6b = await final_decision_agent_6b([taskInfo] + all_thinking_6b[-1], "Sub-task 6b: Synthesize and finalize pattern analysis.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 6b output: thinking - {thinking6b.content}; answer - {answer6b.content}")
    subtask_desc6b['response'] = {"thinking": thinking6b, "answer": answer6b}
    logs.append(subtask_desc6b)
    print("Step 6b: ", sub_tasks[-1])

    reflect_inst_7 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_7 = (
        "Sub-task 7: Count total number of losing positions for Alice (dp[n] == False) among n ≤ 2024 by summing enumerated list from Sub-task 6a. "
        "Verify correctness by cross-checking dp array. Avoid unverified formulas or assumptions. "
        + reflect_inst_7
    )
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_7 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_7 = self.max_round
    cot_inputs_7 = [taskInfo, thinking6a, thinking6b]
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_reflect_instruction_7,
        "context": ["user query", thinking6a.content, thinking6b.content],
        "agent_collaboration": "Reflexion"
    }
    thinking7, answer7 = await cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_7.id}, counting losing positions, thinking: {thinking7.content}; answer: {answer7.content}")
    critic_inst_7 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(N_max_7):
        feedback7, correct7 = await critic_agent_7([taskInfo, thinking7], critic_inst_7, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_7.id}, providing feedback, thinking: {feedback7.content}; answer: {correct7.content}")
        if correct7.content == "True":
            break
        cot_inputs_7.extend([thinking7, feedback7])
        thinking7, answer7 = await cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_7.id}, refining count, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs
