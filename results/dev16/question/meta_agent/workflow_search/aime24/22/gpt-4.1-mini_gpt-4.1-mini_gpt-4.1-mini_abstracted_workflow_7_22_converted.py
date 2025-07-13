async def forward_22(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Formalize problem and constraints (CoT)
    cot_instruction_0_1 = (
        "Sub-task 1: Formally define problem variables: list length n (positive integer), list elements as positive integers, "
        "and their frequencies. Explicitly state constraints: sum of elements = 30, unique mode = 9, median is a positive integer not in the list. "
        "Do not attempt to solve or enumerate solutions yet; focus on formalizing problem and constraints."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, formalizing problem variables and constraints, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task stage_0.subtask_1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 0.1: ", sub_tasks[-1])

    cot_instruction_0_2 = (
        "Sub-task 2: Analyze implications of median being a positive integer not in the list. "
        "Deduce that list length n must be even, median m = (a + b)/2 for two distinct middle elements a < b, "
        "with m integer and m != a, m != b. Avoid assuming specific values or restricting pairs prematurely."
    )
    cot_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_instruction_0_2,
        "context": ["user query", thinking_0_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_2, answer_0_2 = await cot_agent_0_2([taskInfo, thinking_0_1], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_2.id}, analyzing median implications, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Sub-task stage_0.subtask_2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 0.2: ", sub_tasks[-1])

    cot_instruction_0_3 = (
        "Sub-task 3: Analyze unique mode condition: 9 appears more times than any other number. "
        "Formulate constraints on frequency f_9 relative to others, including minimum f_9 >= 2, and no other element matches or exceeds f_9. "
        "Avoid enumerating frequencies yet; focus on logical implications."
    )
    cot_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_3 = {
        "subtask_id": "stage_0.subtask_3",
        "instruction": cot_instruction_0_3,
        "context": ["user query", thinking_0_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_3, answer_0_3 = await cot_agent_0_3([taskInfo, thinking_0_1], cot_instruction_0_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_3.id}, analyzing unique mode constraints, thinking: {thinking_0_3.content}; answer: {answer_0_3.content}")
    sub_tasks.append(f"Sub-task stage_0.subtask_3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)
    print("Step 0.3: ", sub_tasks[-1])

    # Stage 1: Enumerate median pairs (Debate), list lengths (CoT), candidate lists (SC_CoT), and validate (Reflexion)

    cot_instruction_1_1 = (
        "Sub-task 1: Enumerate all feasible even list lengths n in range 4 to 12 consistent with sum=30 and median constraints. "
        "Justify feasibility without fixing element values. Avoid premature pruning."
    )
    cot_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_instruction_1_1,
        "context": [thinking_0_2.content],
        "agent_collaboration": "CoT"
    }
    thinking_1_1, answer_1_1 = await cot_agent_1_1([taskInfo, thinking_0_2], cot_instruction_1_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_1.id}, enumerating feasible list lengths, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Sub-task stage_1.subtask_1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])

    debate_instruction_1_2 = (
        "Sub-task 2: Enumerate all valid pairs of distinct positive integers (a, b) with a < b and (a + b) even, "
        "such that median m = (a + b)/2 is a positive integer not equal to a or b. Enumerate all such pairs up to a reasonable bound (e.g., 1 to 20). "
        "Remove restrictive assumptions about adjacency. Document full set of candidate median pairs. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_1_2 = self.max_round
    all_thinking_1_2 = [[] for _ in range(N_max_1_2)]
    all_answer_1_2 = [[] for _ in range(N_max_1_2)]
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": debate_instruction_1_2,
        "context": [thinking_0_2.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1_2):
        for i, agent in enumerate(debate_agents_1_2):
            if r == 0:
                thinking_1_2, answer_1_2 = await agent([taskInfo, thinking_0_2], debate_instruction_1_2, r, is_sub_task=True)
            else:
                input_infos_1_2 = [taskInfo, thinking_0_2] + all_thinking_1_2[r-1]
                thinking_1_2, answer_1_2 = await agent(input_infos_1_2, debate_instruction_1_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, enumerating median pairs, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
            all_thinking_1_2[r].append(thinking_1_2)
            all_answer_1_2[r].append(answer_1_2)
    final_decision_instruction_1_2 = (
        "Given all the above thinking and answers, reason over them carefully and provide a final answer listing all valid median pairs (a,b) with a < b, (a+b) even, median m = (a+b)/2 integer not equal to a or b."
    )
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo] + all_thinking_1_2[-1], final_decision_instruction_1_2, is_sub_task=True)
    agents.append(f"Final Decision agent, enumerating median pairs, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Sub-task stage_1.subtask_2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1.2: ", sub_tasks[-1])

    cot_sc_instruction_1_3 = (
        "Sub-task 3: Systematically enumerate candidate lists by combining parameters: list length n (from subtask 1), frequency of 9 (f_9 >= 2), and median pairs (a,b) from subtask 2. "
        "Use backtracking or search-based methods to assign frequencies and values to other elements, ensuring sum=30, unique mode=9, and median condition. "
        "Explore broad parameter space without premature pruning."
    )
    N_sc = self.max_sc
    cot_sc_agents_1_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1_3 = []
    possible_thinkings_1_3 = []
    subtask_desc_1_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": cot_sc_instruction_1_3,
        "context": [thinking_1_1.content, thinking_1_2.content, thinking_0_3.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_1_3, answer_1_3 = await cot_sc_agents_1_3[i]([taskInfo, thinking_1_1, thinking_1_2, thinking_0_3], cot_sc_instruction_1_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1_3[i].id}, enumerating candidate lists, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
        possible_answers_1_3.append(answer_1_3)
        possible_thinkings_1_3.append(thinking_1_3)
    final_decision_instruction_1_3 = (
        "Given all the above thinking and answers, synthesize and provide a comprehensive list of candidate lists that satisfy sum=30, unique mode=9, and median conditions. "
        "Include all valid candidates found."
    )
    final_decision_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_3, answer_1_3 = await final_decision_agent_1_3([taskInfo] + possible_thinkings_1_3, final_decision_instruction_1_3, is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing candidate lists, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    sub_tasks.append(f"Sub-task stage_1.subtask_3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)
    print("Step 1.3: ", sub_tasks[-1])

    reflect_instruction_1_4 = (
        "Sub-task 4: Strictly validate each candidate list from subtask 3 against all constraints: sum=30, unique mode=9, median integer not in list, all positive integers. "
        "Discard invalid candidates immediately. Document all valid candidate lists for further analysis. "
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    )
    cot_agent_1_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_1_4 = self.max_round
    cot_inputs_1_4 = [taskInfo, thinking_1_3, answer_1_3]
    subtask_desc_1_4 = {
        "subtask_id": "stage_1.subtask_4",
        "instruction": reflect_instruction_1_4,
        "context": ["user query", thinking_1_3.content, answer_1_3.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_1_4, answer_1_4 = await cot_agent_1_4(cot_inputs_1_4, reflect_instruction_1_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1_4.id}, validating candidate lists, thinking: {thinking_1_4.content}; answer: {answer_1_4.content}")
    for i in range(N_max_1_4):
        feedback_1_4, correct_1_4 = await critic_agent_1_4([taskInfo, thinking_1_4], "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1_4.id}, providing feedback, thinking: {feedback_1_4.content}; answer: {correct_1_4.content}")
        if correct_1_4.content == "True":
            break
        cot_inputs_1_4.extend([thinking_1_4, feedback_1_4])
        thinking_1_4, answer_1_4 = await cot_agent_1_4(cot_inputs_1_4, reflect_instruction_1_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1_4.id}, refining validation, thinking: {thinking_1_4.content}; answer: {answer_1_4.content}")
    sub_tasks.append(f"Sub-task stage_1.subtask_4 output: thinking - {thinking_1_4.content}; answer - {answer_1_4.content}")
    subtask_desc_1_4['response'] = {"thinking": thinking_1_4, "answer": answer_1_4}
    logs.append(subtask_desc_1_4)
    print("Step 1.4: ", sub_tasks[-1])

    # Stage 2: Compute sum of squares (CoT) and analyze results (Reflexion)

    cot_instruction_2_1 = (
        "Sub-task 1: For each validated candidate list from stage_1.subtask_4, compute the sum of the squares of all items accurately. "
        "Avoid recomputing or revalidating constraints at this stage; focus solely on correct calculation."
    )
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_instruction_2_1,
        "context": [thinking_1_4.content, answer_1_4.content],
        "agent_collaboration": "CoT"
    }
    thinking_2_1, answer_2_1 = await cot_agent_2_1([taskInfo, thinking_1_4, answer_1_4], cot_instruction_2_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_1.id}, computing sum of squares, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Sub-task stage_2.subtask_1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2.1: ", sub_tasks[-1])

    reflect_instruction_2_2 = (
        "Sub-task 2: Analyze computed sums of squares to identify the unique or correct solution consistent with all problem constraints. "
        "Confirm solution satisfies sum=30, unique mode=9, and median conditions. If multiple candidates remain, discuss criteria to select correct one or conclude non-uniqueness. "
        "Avoid introducing new assumptions or constraints. Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    )
    cot_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2_2 = self.max_round
    cot_inputs_2_2 = [taskInfo, thinking_2_1, answer_2_1]
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": reflect_instruction_2_2,
        "context": ["user query", thinking_2_1.content, answer_2_1.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, reflect_instruction_2_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, analyzing sums of squares, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    for i in range(N_max_2_2):
        feedback_2_2, correct_2_2 = await critic_agent_2_2([taskInfo, thinking_2_2], "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_2.id}, providing feedback, thinking: {feedback_2_2.content}; answer: {correct_2_2.content}")
        if correct_2_2.content == "True":
            break
        cot_inputs_2_2.extend([thinking_2_2, feedback_2_2])
        thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, reflect_instruction_2_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, refining analysis, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Sub-task stage_2.subtask_2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 2.2: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_2_2, answer_2_2, sub_tasks, agents)
    return final_answer, logs
