async def forward_27(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Formally represent N as a 4-digit number with digits (a,b,c,d), a != 0, "
        "and express the conditions that changing each digit to 1 results in a number divisible by 7 using modular arithmetic. "
        "Derive modular congruences for each digit-change scenario, ensuring precise formulation to avoid ambiguity in digit substitutions and prepare for rigorous modular consistency checks."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    cot_agents_0_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_0_1 = []
    possible_thinkings_0_1 = []
    for i in range(self.max_sc):
        thinking_i, answer_i = await cot_agents_0_1[i]([taskInfo], cot_instruction_0_1, is_sub_task=True)
        agents.append(f"SC_CoT agent {cot_agents_0_1[i].id}, formulating modular conditions, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_0_1.append(answer_i)
        possible_thinkings_0_1.append(thinking_i)
    final_decision_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_1, answer_0_1 = await final_decision_agent_0_1(
        [taskInfo] + possible_thinkings_0_1 + possible_answers_0_1,
        "Sub-task 1: Synthesize and choose the most consistent modular arithmetic conditions for N.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 1: ", sub_tasks[-1])

    debate_instruction_0_2 = (
        "Sub-task 2: Analyze and relate the modular conditions derived in subtask_1 to form a consistent system of congruences that N must satisfy. "
        "Validate the system for internal consistency and prepare it as a foundation for candidate generation. "
        "Ensure the modular system fully captures all digit-change divisibility constraints and is ready for cross-validation with numeric checks. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_0_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking_0_2 = []
    all_answer_0_2 = []
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": debate_instruction_0_2,
        "context": ["user query", thinking_0_1.content, answer_0_1.content],
        "agent_collaboration": "Debate"
    }
    for r in range(self.max_round):
        round_thinking = []
        round_answer = []
        for i, agent in enumerate(debate_agents_0_2):
            if r == 0:
                thinking_i, answer_i = await agent([taskInfo, thinking_0_1, answer_0_1], debate_instruction_0_2, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking_0_1, answer_0_1] + all_thinking_0_2[-1] + all_answer_0_2[-1]
                thinking_i, answer_i = await agent(inputs, debate_instruction_0_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing modular system, thinking: {thinking_i.content}; answer: {answer_i.content}")
            round_thinking.append(thinking_i)
            round_answer.append(answer_i)
        all_thinking_0_2.append(round_thinking)
        all_answer_0_2.append(round_answer)
    final_decision_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await final_decision_agent_0_2(
        [taskInfo, thinking_0_1, answer_0_1] + all_thinking_0_2[-1] + all_answer_0_2[-1],
        "Sub-task 2: Synthesize and finalize the consistent modular system for candidate generation.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 2: ", sub_tasks[-1])

    debate_instruction_1_1 = (
        "Sub-task 1: Enumerate all candidate digit tuples (a,b,c,d) that satisfy the modular arithmetic system derived in stage_0.subtask_2. "
        "Ensure candidates strictly satisfy modular residue conditions before any divisibility testing, preventing premature acceptance of invalid candidates. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking_1_1 = []
    all_answer_1_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": debate_instruction_1_1,
        "context": ["user query", thinking_0_2.content, answer_0_2.content],
        "agent_collaboration": "Debate"
    }
    for r in range(self.max_round):
        round_thinking = []
        round_answer = []
        for i, agent in enumerate(debate_agents_1_1):
            if r == 0:
                thinking_i, answer_i = await agent([taskInfo, thinking_0_2, answer_0_2], debate_instruction_1_1, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking_0_2, answer_0_2] + all_thinking_1_1[-1] + all_answer_1_1[-1]
                thinking_i, answer_i = await agent(inputs, debate_instruction_1_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, enumerating candidates, thinking: {thinking_i.content}; answer: {answer_i.content}")
            round_thinking.append(thinking_i)
            round_answer.append(answer_i)
        all_thinking_1_1.append(round_thinking)
        all_answer_1_1.append(round_answer)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1(
        [taskInfo, thinking_0_2, answer_0_2] + all_thinking_1_1[-1] + all_answer_1_1[-1],
        "Sub-task 1: Finalize candidate digit tuples satisfying modular conditions.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 3: ", sub_tasks[-1])

    debate_instruction_1_2 = (
        "Sub-task 2: Implement an explicit, automated divisibility-check subtask that programmatically tests each candidate N and all four digit-change variants (changing each digit to 1) for divisibility by 7. "
        "Cross-validate modular arithmetic constraints with numeric divisibility to eliminate human arithmetic errors and false positives. "
        "Reject any candidate failing any divisibility test, preventing acceptance of invalid candidates like 9435. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking_1_2 = []
    all_answer_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": debate_instruction_1_2,
        "context": ["user query", thinking_1_1.content, answer_1_1.content],
        "agent_collaboration": "Debate"
    }
    for r in range(self.max_round):
        round_thinking = []
        round_answer = []
        for i, agent in enumerate(debate_agents_1_2):
            if r == 0:
                thinking_i, answer_i = await agent([taskInfo, thinking_1_1, answer_1_1], debate_instruction_1_2, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking_1_1, answer_1_1] + all_thinking_1_2[-1] + all_answer_1_2[-1]
                thinking_i, answer_i = await agent(inputs, debate_instruction_1_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, divisibility check, thinking: {thinking_i.content}; answer: {answer_i.content}")
            round_thinking.append(thinking_i)
            round_answer.append(answer_i)
        all_thinking_1_2.append(round_thinking)
        all_answer_1_2.append(round_answer)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2(
        [taskInfo, thinking_1_1, answer_1_1] + all_thinking_1_2[-1] + all_answer_1_2[-1],
        "Sub-task 2: Finalize candidates passing all divisibility tests.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 4: ", sub_tasks[-1])

    debate_instruction_1_3 = (
        "Sub-task 3: Perform a final brute-force enumeration from 9999 down to 1000, applying the automated divisibility checks from subtask_2 to each candidate, "
        "and immediately select the greatest valid N that satisfies all conditions. "
        "Ensure correctness and completeness, addressing feedback that previous attempts failed due to insufficient verification and premature candidate acceptance. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_1_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking_1_3 = []
    all_answer_1_3 = []
    subtask_desc_1_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": debate_instruction_1_3,
        "context": ["user query", thinking_1_2.content, answer_1_2.content],
        "agent_collaboration": "Debate"
    }
    for r in range(self.max_round):
        round_thinking = []
        round_answer = []
        for i, agent in enumerate(debate_agents_1_3):
            if r == 0:
                thinking_i, answer_i = await agent([taskInfo, thinking_1_2, answer_1_2], debate_instruction_1_3, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking_1_2, answer_1_2] + all_thinking_1_3[-1] + all_answer_1_3[-1]
                thinking_i, answer_i = await agent(inputs, debate_instruction_1_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, brute-force search, thinking: {thinking_i.content}; answer: {answer_i.content}")
            round_thinking.append(thinking_i)
            round_answer.append(answer_i)
        all_thinking_1_3.append(round_thinking)
        all_answer_1_3.append(round_answer)
    final_decision_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_3, answer_1_3 = await final_decision_agent_1_3(
        [taskInfo, thinking_1_2, answer_1_2] + all_thinking_1_3[-1] + all_answer_1_3[-1],
        "Sub-task 3: Select the greatest valid N after brute-force verification.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)
    print("Step 5: ", sub_tasks[-1])

    reflect_instruction_2_1 = (
        "Sub-task 1: Decompose the identified greatest valid number N into quotient Q and remainder R upon division by 1000, i.e., find Q and R such that N = 1000Q + R. "
        "This subtask depends on the verified candidate from stage_1.subtask_3 to ensure correctness of subsequent calculations. "
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    )
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_2_1 = [taskInfo, thinking_1_3, answer_1_3]
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": reflect_instruction_2_1,
        "context": ["user query", thinking_1_3.content, answer_1_3.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_1, answer_2_1 = await cot_agent_2_1(cot_inputs_2_1, reflect_instruction_2_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_1.id}, decomposing N into Q and R, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    for i in range(self.max_round):
        feedback_2_1, correct_2_1 = await critic_agent_2_1(
            [taskInfo, thinking_2_1, answer_2_1],
            "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'",
            i, is_sub_task=True
        )
        agents.append(f"Critic agent {critic_agent_2_1.id}, providing feedback, thinking: {feedback_2_1.content}; answer: {correct_2_1.content}")
        if correct_2_1.content == "True":
            break
        cot_inputs_2_1.extend([thinking_2_1, answer_2_1, feedback_2_1])
        thinking_2_1, answer_2_1 = await cot_agent_2_1(cot_inputs_2_1, reflect_instruction_2_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_1.id}, refining decomposition, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 6: ", sub_tasks[-1])

    cot_instruction_3_1 = (
        "Sub-task 1: Compute the sum Q + R and present the final result as required by the problem. "
        "This subtask consolidates all prior verified results and ensures the final output is consistent with the problem statement."
    )
    cot_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_3_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": cot_instruction_3_1,
        "context": ["user query", thinking_2_1.content, answer_2_1.content],
        "agent_collaboration": "SC_CoT"
    }
    cot_agents_3_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_3_1 = []
    possible_thinkings_3_1 = []
    for i in range(self.max_sc):
        thinking_i, answer_i = await cot_agents_3_1[i]([taskInfo, thinking_2_1, answer_2_1], cot_instruction_3_1, is_sub_task=True)
        agents.append(f"SC_CoT agent {cot_agents_3_1[i].id}, computing Q+R, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_3_1.append(answer_i)
        possible_thinkings_3_1.append(thinking_i)
    final_decision_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3_1, answer_3_1 = await final_decision_agent_3_1(
        [taskInfo, thinking_2_1, answer_2_1] + possible_thinkings_3_1 + possible_answers_3_1,
        "Sub-task 1: Synthesize and finalize the sum Q+R as the final answer.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3_1, answer_3_1, sub_tasks, agents)
    return final_answer, logs
