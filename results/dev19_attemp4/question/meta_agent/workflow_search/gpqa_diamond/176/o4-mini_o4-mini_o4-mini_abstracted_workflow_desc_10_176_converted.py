async def forward_176(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0, Sub-task 0.1: Extract parameters
    cot_instruction = (
        "Sub-task 0.1: Extract the radius ratio, mass ratio, observed peak wavelengths, radial velocities, and answer choices from the query."  
    )
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent",
                              model=self.node_model, temperature=0.0)
    subtask_desc = {
        "subtask_id": "stage0_subtask1",
        "instruction": cot_instruction,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking0_1, answer0_1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(
        f"CoT agent {cot_agent.id}, extracted parameters, thinking: {thinking0_1.content}; answer: {answer0_1.content}"
    )
    sub_tasks.append(f"Sub-task 0.1 output: thinking - {thinking0_1.content}; answer - {answer0_1.content}")
    subtask_desc['response'] = {"thinking": thinking0_1, "answer": answer0_1}
    logs.append(subtask_desc)
    print("Step 1: ", sub_tasks[-1])

    # Stage 0, Sub-task 0.2: Doppler correction of peak wavelengths
    sc_instruction = (
        "Sub-task 0.2: Apply Doppler shift corrections to the observed peak wavelengths for both stars to get rest-frame values."  
    )
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent",
                                model=self.node_model, temperature=0.5)
                  for _ in range(N)]
    possible_thinkings = []
    possible_answers = []
    subtask_desc = {
        "subtask_id": "stage0_subtask2",
        "instruction": sc_instruction,
        "context": ["user query", "thinking of subtask 0.1", "answer of subtask 0.1"],
        "agent_collaboration": "SC_CoT"
    }
    for i, agent in enumerate(cot_agents):
        thinking, answer = await agent([taskInfo, thinking0_1, answer0_1], sc_instruction, is_sub_task=True)
        agents.append(
            f"CoT-SC agent {agent.id}, applied Doppler corrections, thinking: {thinking.content}; answer: {answer.content}"
        )
        possible_thinkings.append(thinking)
        possible_answers.append(answer)
    final_decision_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent",
                                        model=self.node_model, temperature=0.0)
    final_inst = (
        "Sub-task 0.2 final: Given all the above thinking and answers, synthesize and choose the most consistent rest-frame peak wavelengths."  
    )
    thinking0_2, answer0_2 = await final_decision_agent(
        [taskInfo, thinking0_1, answer0_1] + possible_thinkings + possible_answers,
        final_inst, is_sub_task=True
    )
    sub_tasks.append(
        f"Sub-task 0.2 output: thinking - {thinking0_2.content}; answer - {answer0_2.content}"
    )
    subtask_desc['response'] = {"thinking": thinking0_2, "answer": answer0_2}
    logs.append(subtask_desc)
    print("Step 2: ", sub_tasks[-1])

    # Stage 1, Sub-task 1.1: Compute temperatures via Wien's law
    cot_instruction = (
        "Sub-task 1.1: Use Wien’s displacement law on the corrected rest-frame peak wavelengths to compute each star’s effective temperature and confirm their ratio."  
    )
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent",
                              model=self.node_model, temperature=0.0)
    subtask_desc = {
        "subtask_id": "stage1_subtask1",
        "instruction": cot_instruction,
        "context": ["user query", "thinking of subtask 0.2", "answer of subtask 0.2"],
        "agent_collaboration": "CoT"
    }
    thinking1_1, answer1_1 = await cot_agent([taskInfo, thinking0_2, answer0_2], cot_instruction, is_sub_task=True)
    agents.append(
        f"CoT agent {cot_agent.id}, computed temperatures, thinking: {thinking1_1.content}; answer: {answer1_1.content}"
    )
    sub_tasks.append(
        f"Sub-task 1.1 output: thinking - {thinking1_1.content}; answer - {answer1_1.content}"
    )
    subtask_desc['response'] = {"thinking": thinking1_1, "answer": answer1_1}
    logs.append(subtask_desc)
    print("Step 3: ", sub_tasks[-1])

    # Stage 1, Sub-task 1.2: Calculate luminosity ratio
    sc_instruction = (
        "Sub-task 1.2: Calculate the luminosity ratio L1/L2 using L ∝ R² T⁴, with radius ratio and temperature ratio from previous outputs."  
    )
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent",
                                model=self.node_model, temperature=0.5)
                  for _ in range(N)]
    possible_thinkings = []
    possible_answers = []
    subtask_desc = {
        "subtask_id": "stage1_subtask2",
        "instruction": sc_instruction,
        "context": ["user query", "thinking of subtask 0.1", "answer of subtask 0.1", "thinking of subtask 1.1", "answer of subtask 1.1"],
        "agent_collaboration": "SC_CoT"
    }
    for i, agent in enumerate(cot_agents):
        thinking, answer = await agent(
            [taskInfo, thinking0_1, answer0_1, thinking1_1, answer1_1],
            sc_instruction, is_sub_task=True
        )
        agents.append(
            f"CoT-SC agent {agent.id}, calculated luminosity ratio, thinking: {thinking.content}; answer: {answer.content}"
        )
        possible_thinkings.append(thinking)
        possible_answers.append(answer)
    final_decision_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent",
                                        model=self.node_model, temperature=0.0)
    final_inst = (
        "Sub-task 1.2 final: Given all above thinking and answers, synthesize and choose the most consistent luminosity ratio."  
    )
    thinking1_2, answer1_2 = await final_decision_agent(
        [taskInfo, thinking0_1, answer0_1, thinking1_1, answer1_1] + possible_thinkings + possible_answers,
        final_inst, is_sub_task=True
    )
    sub_tasks.append(
        f"Sub-task 1.2 output: thinking - {thinking1_2.content}; answer - {answer1_2.content}"
    )
    subtask_desc['response'] = {"thinking": thinking1_2, "answer": answer1_2}
    logs.append(subtask_desc)
    print("Step 4: ", sub_tasks[-1])

    # Stage 2, Sub-task 2.1: Debate to select final answer
    debate_instruction = (
        "Sub-task 2.1: Compare the computed luminosity ratio to the provided choices and select the closest match as the final answer. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", role=role, model=self.node_model, temperature=0.5)
                     for role in self.debate_role]
    all_thinking = [[] for _ in range(self.max_round)]
    all_answer = [[] for _ in range(self.max_round)]
    subtask_desc = {
        "subtask_id": "stage2_subtask1",
        "instruction": debate_instruction,
        "context": ["user query", "thinking of subtask 1.2", "answer of subtask 1.2"],
        "agent_collaboration": "Debate"
    }
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking1_2, answer1_2], debate_instruction, r, is_sub_task=True)
            else:
                thinking, answer = await agent(
                    [taskInfo, thinking1_2, answer1_2] + all_thinking[r-1] + all_answer[r-1],
                    debate_instruction, r, is_sub_task=True
                )
            agents.append(
                f"Debate agent {agent.id}, round {r}, thinking: {thinking.content}; answer: {answer.content}"
            )
            all_thinking[r].append(thinking)
            all_answer[r].append(answer)
    final_decision_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent",
                                        model=self.node_model, temperature=0.0)
    final_inst = (
        "Sub-task 2.1 final: Given all the above thinking and answers, reason over them carefully and provide a final answer."
    )
    thinking2_1, answer2_1 = await final_decision_agent(
        [taskInfo, thinking1_2, answer1_2] + all_thinking[-1] + all_answer[-1],
        final_inst, is_sub_task=True
    )
    agents.append(
        f"Final Decision agent {final_decision_agent.id}, thinking: {thinking2_1.content}; answer: {answer2_1.content}"
    )
    sub_tasks.append(
        f"Sub-task 2.1 output: thinking - {thinking2_1.content}; answer - {answer2_1.content}"
    )
    subtask_desc['response'] = {"thinking": thinking2_1, "answer": answer2_1}
    logs.append(subtask_desc)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking2_1, answer2_1, sub_tasks, agents)
    return final_answer, logs
