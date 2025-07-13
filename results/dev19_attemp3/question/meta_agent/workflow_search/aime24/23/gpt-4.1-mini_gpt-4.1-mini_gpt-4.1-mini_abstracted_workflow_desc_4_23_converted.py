async def forward_23(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction_0 = (
        "Sub-task 1: Derive and validate formal algebraic expressions representing the sum constraints for the rows and columns of the 2x3 digit grid, "
        "including the digit-level complement relation N2 = 999 - N1 which implies (d,e,f) = (9 - a, 9 - b, 9 - c). "
        "Explicitly confirm place value representations, the example provided, and clarify leading zeros and digit ranges."
    )
    cot_sc_agent_0 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_0 = []
    possible_thinkings_0 = []
    subtask_desc_0 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_sc_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking0, answer0 = await cot_sc_agent_0[i]([taskInfo], cot_sc_instruction_0, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agent_0[i].id}, deriving algebraic relations, thinking: {thinking0.content}; answer: {answer0.content}")
        possible_answers_0.append(answer0)
        possible_thinkings_0.append(thinking0)
    final_decision_agent_0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking0, answer0 = await final_decision_agent_0([taskInfo] + possible_thinkings_0 + possible_answers_0, "Sub-task 1: Synthesize and confirm algebraic complement relation and place value expressions.", is_sub_task=True)
    sub_tasks.append(f"Stage 0 Sub-task 1 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc_0['response'] = {"thinking": thinking0, "answer": answer0}
    logs.append(subtask_desc_0)
    print("Step 0: ", sub_tasks[-1])

    debate_instruction_1_1 = (
        "Sub-task 1: Enumerate all possible digit triples (a,b,c) for the first row such that the sum of digits S = a + b + c satisfies the row sum constraint and place value constraints. "
        "Avoid overcounting or undercounting and prepare for complement computation. Given solutions from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_instruction_1_2 = (
        "Sub-task 2: Enumerate all possible digit triples (d,e,f) for the second row consistent with the complement relation (d,e,f) = (9 - a, 9 - b, 9 - c). "
        "Prepare for cross-verification with first row digits. Given solutions from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_instruction_1_3 = (
        "Sub-task 3: Enumerate all possible digit triples (a,b,c) and (d,e,f) that satisfy the column sum constraint (sum of three 2-digit numbers equals 99), "
        "deriving possible values of S and T from 10S + T = 99. Identify all candidate (S,T) pairs before applying the complement relation. Given solutions from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )

    debate_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    debate_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    debate_agents_1_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]

    N_max_1 = self.max_round

    all_thinking_1_1 = [[] for _ in range(N_max_1)]
    all_answer_1_1 = [[] for _ in range(N_max_1)]
    all_thinking_1_2 = [[] for _ in range(N_max_1)]
    all_answer_1_2 = [[] for _ in range(N_max_1)]
    all_thinking_1_3 = [[] for _ in range(N_max_1)]
    all_answer_1_3 = [[] for _ in range(N_max_1)]

    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": debate_instruction_1_1,
        "context": ["user query", thinking0, answer0],
        "agent_collaboration": "Debate"
    }
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": debate_instruction_1_2,
        "context": ["user query", thinking0, answer0],
        "agent_collaboration": "Debate"
    }
    subtask_desc_1_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": debate_instruction_1_3,
        "context": ["user query", thinking0, answer0],
        "agent_collaboration": "Debate"
    }

    for r in range(N_max_1):
        for i, agent in enumerate(debate_agents_1_1):
            if r == 0:
                thinking1_1, answer1_1 = await agent([taskInfo, thinking0, answer0], debate_instruction_1_1, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking0, answer0] + all_thinking_1_1[r-1] + all_answer_1_1[r-1]
                thinking1_1, answer1_1 = await agent(input_infos, debate_instruction_1_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, enumerating first row triples, thinking: {thinking1_1.content}; answer: {answer1_1.content}")
            all_thinking_1_1[r].append(thinking1_1)
            all_answer_1_1[r].append(answer1_1)

        for i, agent in enumerate(debate_agents_1_2):
            if r == 0:
                thinking1_2, answer1_2 = await agent([taskInfo, thinking0, answer0], debate_instruction_1_2, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking0, answer0] + all_thinking_1_2[r-1] + all_answer_1_2[r-1]
                thinking1_2, answer1_2 = await agent(input_infos, debate_instruction_1_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, enumerating second row triples, thinking: {thinking1_2.content}; answer: {answer1_2.content}")
            all_thinking_1_2[r].append(thinking1_2)
            all_answer_1_2[r].append(answer1_2)

        for i, agent in enumerate(debate_agents_1_3):
            if r == 0:
                thinking1_3, answer1_3 = await agent([taskInfo, thinking0, answer0], debate_instruction_1_3, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking0, answer0] + all_thinking_1_3[r-1] + all_answer_1_3[r-1]
                thinking1_3, answer1_3 = await agent(input_infos, debate_instruction_1_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, enumerating candidate (S,T) pairs, thinking: {thinking1_3.content}; answer: {answer1_3.content}")
            all_thinking_1_3[r].append(thinking1_3)
            all_answer_1_3[r].append(answer1_3)

    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1_1, answer1_1 = await final_decision_agent_1_1([taskInfo, thinking0, answer0] + all_thinking_1_1[-1] + all_answer_1_1[-1], "Sub-task 1: Finalize enumeration of first row digit triples.", is_sub_task=True)
    sub_tasks.append(f"Stage 1 Sub-task 1 output: thinking - {thinking1_1.content}; answer - {answer1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking1_1, "answer": answer1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])

    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1_2, answer1_2 = await final_decision_agent_1_2([taskInfo, thinking0, answer0] + all_thinking_1_2[-1] + all_answer_1_2[-1], "Sub-task 2: Finalize enumeration of second row digit triples.", is_sub_task=True)
    sub_tasks.append(f"Stage 1 Sub-task 2 output: thinking - {thinking1_2.content}; answer - {answer1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking1_2, "answer": answer1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1.2: ", sub_tasks[-1])

    final_decision_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1_3, answer1_3 = await final_decision_agent_1_3([taskInfo, thinking0, answer0] + all_thinking_1_3[-1] + all_answer_1_3[-1], "Sub-task 3: Finalize enumeration of candidate (S,T) pairs from column sum constraint.", is_sub_task=True)
    sub_tasks.append(f"Stage 1 Sub-task 3 output: thinking - {thinking1_3.content}; answer - {answer1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking1_3, "answer": answer1_3}
    logs.append(subtask_desc_1_3)
    print("Step 1.3: ", sub_tasks[-1])

    debate_instruction_2_1 = (
        "Sub-task 1: Apply the digit-level complement relation (d,e,f) = (9 - a, 9 - b, 9 - c) derived from N2 = 999 - N1 to filter candidate (S,T) pairs obtained from the column sum constraint. "
        "Enforce the identity T = 27 - S and eliminate invalid (S,T) pairs before enumeration, preventing double counting and invalid solutions. Given solutions from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    reflexion_instruction_2_2 = (
        "Sub-task 2: Decompose combined constraints from row and column sums, incorporating the complement relation, to simplify the search space for valid digit assignments. "
        "Analyze digit sum ranges, place value constraints, and dependencies between digits to prepare for systematic enumeration. Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    )

    debate_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    cot_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)

    N_max_2_1 = self.max_round

    all_thinking_2_1 = [[] for _ in range(N_max_2_1)]
    all_answer_2_1 = [[] for _ in range(N_max_2_1)]

    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": debate_instruction_2_1,
        "context": ["user query", thinking1_3, answer1_3, thinking1_1, answer1_1, thinking1_2, answer1_2],
        "agent_collaboration": "Debate"
    }

    for r in range(N_max_2_1):
        for i, agent in enumerate(debate_agents_2_1):
            if r == 0:
                thinking2_1, answer2_1 = await agent([taskInfo, thinking1_3, answer1_3, thinking1_1, answer1_1, thinking1_2, answer1_2], debate_instruction_2_1, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking1_3, answer1_3, thinking1_1, answer1_1, thinking1_2, answer1_2] + all_thinking_2_1[r-1] + all_answer_2_1[r-1]
                thinking2_1, answer2_1 = await agent(input_infos, debate_instruction_2_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, filtering candidate (S,T) pairs, thinking: {thinking2_1.content}; answer: {answer2_1.content}")
            all_thinking_2_1[r].append(thinking2_1)
            all_answer_2_1[r].append(answer2_1)

    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2_1, answer2_1 = await final_decision_agent_2_1([taskInfo, thinking1_3, answer1_3, thinking1_1, answer1_1, thinking1_2, answer1_2] + all_thinking_2_1[-1] + all_answer_2_1[-1], "Sub-task 1: Finalize filtering of candidate (S,T) pairs with complement relation.", is_sub_task=True)
    sub_tasks.append(f"Stage 2 Sub-task 1 output: thinking - {thinking2_1.content}; answer - {answer2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking2_1, "answer": answer2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2.1: ", sub_tasks[-1])

    cot_reflect_instruction_2_2 = (
        "Sub-task 2: Decompose combined constraints from row and column sums, incorporating the complement relation, to simplify the search space for valid digit assignments. "
        "Analyze digit sum ranges, place value constraints, and dependencies between digits to prepare for systematic enumeration. "
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    )

    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_reflect_instruction_2_2,
        "context": ["user query", thinking2_1, answer2_1],
        "agent_collaboration": "Reflexion"
    }

    thinking2_2, answer2_2 = await cot_agent_2_2([taskInfo, thinking2_1, answer2_1], cot_reflect_instruction_2_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, decomposing constraints, thinking: {thinking2_2.content}; answer: {answer2_2.content}")

    for i in range(self.max_round):
        feedback, correct = await critic_agent_2_2([taskInfo, thinking2_2, answer2_2], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_2.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip() == "True":
            break
        thinking2_2, answer2_2 = await cot_agent_2_2([taskInfo, thinking2_1, answer2_1, thinking2_2, answer2_2, feedback], cot_reflect_instruction_2_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, refining decomposition, thinking: {thinking2_2.content}; answer: {answer2_2.content}")

    sub_tasks.append(f"Stage 2 Sub-task 2 output: thinking - {thinking2_2.content}; answer - {answer2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking2_2, "answer": answer2_2}
    logs.append(subtask_desc_2_2)
    print("Step 2.2: ", sub_tasks[-1])

    cot_sc_instruction_3_1 = (
        "Sub-task 1: Systematically enumerate all valid digit assignments (a,b,c,d,e,f) satisfying both the row sum and column sum constraints simultaneously, "
        "using the filtered (S,T) pairs from stage_2.subtask_1. For each candidate triple (a,b,c) with sum S, compute N1 = 100a + 10b + c, then compute N2 = 999 - N1 and decompose into digits (d,e,f). "
        "Verify that (d,e,f) = (9 - a, 9 - b, 9 - c), that digit sums match T, and that all digits are valid (0-9). Count only valid assignments to avoid double counting or invalid solutions."
    )

    cot_sc_agents_3_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_3_1 = []
    possible_thinkings_3_1 = []
    subtask_desc_3_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": cot_sc_instruction_3_1,
        "context": ["user query", thinking2_2, answer2_2, thinking0, answer0],
        "agent_collaboration": "SC_CoT"
    }

    for i in range(self.max_sc):
        thinking3_1, answer3_1 = await cot_sc_agents_3_1[i]([taskInfo, thinking2_2, answer2_2, thinking0, answer0], cot_sc_instruction_3_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_3_1[i].id}, enumerating valid digit assignments, thinking: {thinking3_1.content}; answer: {answer3_1.content}")
        possible_answers_3_1.append(answer3_1)
        possible_thinkings_3_1.append(thinking3_1)

    final_decision_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3_1, answer3_1 = await final_decision_agent_3_1([taskInfo, thinking2_2, answer2_2, thinking0, answer0] + possible_thinkings_3_1 + possible_answers_3_1, "Sub-task 1: Synthesize and count all valid digit assignments satisfying all constraints.", is_sub_task=True)
    sub_tasks.append(f"Stage 3 Sub-task 1 output: thinking - {thinking3_1.content}; answer - {answer3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking3_1, "answer": answer3_1}
    logs.append(subtask_desc_3_1)
    print("Step 3.1: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking3_1, answer3_1, sub_tasks, agents)
    return final_answer, logs
