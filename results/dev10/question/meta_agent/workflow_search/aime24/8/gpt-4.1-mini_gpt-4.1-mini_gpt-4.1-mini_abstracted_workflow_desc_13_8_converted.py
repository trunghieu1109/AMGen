async def forward_8(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Derive the formal definitions of winning and losing positions for the game where a player can remove 1 or 4 tokens, "
        "and the player who removes the last token wins. Define base cases such as n=0 and n=1, and recursively characterize positions as winning or losing depending on whether there exists a move to a losing position. "
        "Validate these definitions with small values of n to ensure correctness.")
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, deriving winning/losing definitions, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)

    cot_sc_instruction_0_2 = (
        "Sub-task 2: Based on the definitions from Sub-task 1, enumerate the winning and losing status for n=1 to n=10. "
        "Confirm consistency with the recursive definitions and game rules to ensure correct identification of losing positions for the first player (Alice)."
    )
    N_sc = self.max_sc
    cot_agents_0_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_0_2 = []
    possible_thinkings_0_2 = []
    subtask_desc_0_2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_0_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_0_2[i]([taskInfo, thinking_0_1, answer_0_1], cot_sc_instruction_0_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_2[i].id}, enumerating winning/losing for n=1..10, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_0_2.append(answer_i)
        possible_thinkings_0_2.append(thinking_i)
    final_decision_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await final_decision_agent_0_2([taskInfo] + possible_answers_0_2 + possible_thinkings_0_2, 
                                                              "Sub-task 2: Synthesize and choose the most consistent answer for enumerating winning/losing positions for n=1 to 10.", 
                                                              is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 1: ", sub_tasks[-1])

    reflect_inst_1_1 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_1_1 = (
        "Sub-task 1: Analyze the pattern of winning and losing positions identified in Stage 0. "
        "Derive a composite measure or closed-form characterization (such as modular arithmetic conditions) that succinctly describes losing positions for the first player. "
        "Explore if losing positions occur periodically or follow a known sequence. " + reflect_inst_1_1
    )
    cot_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_1_1 = self.max_round
    cot_inputs_1_1 = [taskInfo, thinking_0_2, answer_0_2]
    subtask_desc_1_1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_reflect_instruction_1_1,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Reflexion | CoT"
    }
    thinking_1_1, answer_1_1 = await cot_agent_1_1(cot_inputs_1_1, cot_reflect_instruction_1_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1_1.id}, analyzing pattern of winning/losing positions, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    for i in range(N_max_1_1):
        critic_inst_1_1 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
        feedback_1_1, correct_1_1 = await critic_agent_1_1([taskInfo, thinking_1_1, answer_1_1], 
                                                          "Please review and provide the limitations of provided solutions." + critic_inst_1_1, 
                                                          i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1_1.id}, providing feedback, thinking: {feedback_1_1.content}; answer: {correct_1_1.content}")
        if correct_1_1.content == "True":
            break
        cot_inputs_1_1.extend([thinking_1_1, answer_1_1, feedback_1_1])
        thinking_1_1, answer_1_1 = await cot_agent_1_1(cot_inputs_1_1, cot_reflect_instruction_1_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1_1.id}, refining pattern analysis, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 2: ", sub_tasks[-1])

    cot_instruction_2_1 = (
        "Sub-task 1: Infer explicit parameters or formulas from the pattern found in Stage 1. "
        "For example, determine the modulus and residue class that characterize losing positions. "
        "Compute or prove the formula rigorously to confirm it holds for all n, not just small tested values."
    )
    N_sc_2_1 = self.max_sc
    cot_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_2_1)]
    possible_answers_2_1 = []
    possible_thinkings_2_1 = []
    subtask_desc_2_1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_2_1,
        "context": ["user query", "thinking of subtask 1 stage 1", "answer of subtask 1 stage 1"],
        "agent_collaboration": "CoT | SC_CoT"
    }
    for i in range(N_sc_2_1):
        thinking_i, answer_i = await cot_agents_2_1[i]([taskInfo, thinking_1_1, answer_1_1], cot_instruction_2_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_1[i].id}, inferring formula for losing positions, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_2_1.append(answer_i)
        possible_thinkings_2_1.append(thinking_i)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo] + possible_answers_2_1 + possible_thinkings_2_1, 
                                                              "Sub-task 1: Synthesize and confirm formula for losing positions.", 
                                                              is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 3: ", sub_tasks[-1])

    cot_sc_instruction_3_1 = (
        "Sub-task 1: Using the formula or pattern from Stage 2, enumerate all integers n ≤ 2024 for which the initial position is losing for Alice (thus winning for Bob). "
        "Count these values and provide the final answer. Verify correctness by cross-checking with earlier enumerations and logical consistency."
    )
    N_sc_3_1 = self.max_sc
    cot_agents_3_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_3_1)]
    possible_answers_3_1 = []
    possible_thinkings_3_1 = []
    subtask_desc_3_1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction_3_1,
        "context": ["user query", "thinking of subtask 1 stage 2", "answer of subtask 1 stage 2"],
        "agent_collaboration": "SC_CoT | CoT"
    }
    for i in range(N_sc_3_1):
        thinking_i, answer_i = await cot_agents_3_1[i]([taskInfo, thinking_2_1, answer_2_1], cot_sc_instruction_3_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3_1[i].id}, enumerating losing positions up to 2024, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_3_1.append(answer_i)
        possible_thinkings_3_1.append(thinking_i)
    final_decision_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3_1, answer_3_1 = await final_decision_agent_3_1([taskInfo] + possible_answers_3_1 + possible_thinkings_3_1, 
                                                              "Sub-task 1: Synthesize and count all losing positions for n ≤ 2024.", 
                                                              is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3_1, answer_3_1, sub_tasks, agents)
    return final_answer, logs
