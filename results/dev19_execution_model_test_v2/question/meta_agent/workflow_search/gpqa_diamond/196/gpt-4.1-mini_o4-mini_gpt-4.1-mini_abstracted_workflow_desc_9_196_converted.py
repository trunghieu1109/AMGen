async def forward_196(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    debate_instr_stage0 = "Sub-task 1: Extract and summarize defining spectral features of Compound X from the given IR and 1H NMR data, identifying key functional groups and structural clues. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_stage0 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_stage0 = self.max_round
    all_thinking_stage0 = [[] for _ in range(N_max_stage0)]
    all_answer_stage0 = [[] for _ in range(N_max_stage0)]
    subtask_desc_stage0 = {
        "subtask_id": "subtask_1",
        "instruction": debate_instr_stage0,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_stage0):
        for i, agent in enumerate(debate_agents_stage0):
            if r == 0:
                thinking, answer = await agent([taskInfo], debate_instr_stage0, r, is_sub_task=True)
            else:
                input_infos = [taskInfo] + all_thinking_stage0[r-1] + all_answer_stage0[r-1]
                thinking, answer = await agent(input_infos, debate_instr_stage0, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing spectral data, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_stage0[r].append(thinking)
            all_answer_stage0[r].append(answer)
    final_decision_agent_stage0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking0, answer0 = await final_decision_agent_stage0([taskInfo] + all_thinking_stage0[-1] + all_answer_stage0[-1], "Sub-task 1: Extract and summarize spectral features. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent stage 0, thinking: {thinking0.content}; answer: {answer0.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc_stage0['response'] = {"thinking": thinking0, "answer": answer0}
    logs.append(subtask_desc_stage0)
    print("Step 0: ", sub_tasks[-1])

    cot_sc_instruction_stage1 = "Sub-task 1: Analyze the chemical reaction of Compound X with red phosphorus and HI, determining the likely transformation(s) occurring to functional groups or substituents, based on the output from Sub-task 1."
    N_sc = self.max_sc
    cot_agents_stage1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_stage1 = []
    possible_thinkings_stage1 = []
    subtask_desc_stage1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction_stage1,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking1, answer1 = await cot_agents_stage1[i]([taskInfo, thinking0, answer0], cot_sc_instruction_stage1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_stage1[i].id}, analyzing reaction with red phosphorus and HI, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_answers_stage1.append(answer1)
        possible_thinkings_stage1.append(thinking1)
    final_decision_agent_stage1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_stage1([taskInfo, thinking0, answer0] + possible_thinkings_stage1 + possible_answers_stage1, "Sub-task 2: Synthesize and choose the most consistent answer for reaction analysis.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc_stage1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc_stage1)
    print("Step 1: ", sub_tasks[-1])

    reflect_inst_stage2 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_stage2 = "Sub-task 1: Integrate the spectral characterization of Compound X with the predicted chemical transformation to deduce the structure of the final product." + reflect_inst_stage2
    cot_agent_stage2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_stage2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_stage2 = self.max_round
    cot_inputs_stage2 = [taskInfo, thinking0, answer0, thinking1, answer1]
    subtask_desc_stage2 = {
        "subtask_id": "subtask_1",
        "instruction": cot_reflect_instruction_stage2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Reflexion"
    }
    thinking2, answer2 = await cot_agent_stage2(cot_inputs_stage2, cot_reflect_instruction_stage2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_stage2.id}, initial integration, thinking: {thinking2.content}; answer: {answer2.content}")
    critic_inst_stage2 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(N_max_stage2):
        feedback, correct = await critic_agent_stage2([taskInfo, thinking2, answer2], "Please review and provide the limitations of provided solutions." + critic_inst_stage2, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_stage2.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip() == "True":
            break
        cot_inputs_stage2.extend([thinking2, answer2, feedback])
        thinking2, answer2 = await cot_agent_stage2(cot_inputs_stage2, cot_reflect_instruction_stage2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_stage2.id}, refining integration, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc_stage2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc_stage2)
    print("Step 2: ", sub_tasks[-1])

    debate_instr_stage3 = "Sub-task 1: Evaluate and prioritize the given candidate structures against the integrated spectral and reaction analysis to identify the most plausible final product. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_stage3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_stage3 = self.max_round
    all_thinking_stage3 = [[] for _ in range(N_max_stage3)]
    all_answer_stage3 = [[] for _ in range(N_max_stage3)]
    subtask_desc_stage3 = {
        "subtask_id": "subtask_1",
        "instruction": debate_instr_stage3,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_stage3):
        for i, agent in enumerate(debate_agents_stage3):
            if r == 0:
                thinking3, answer3 = await agent([taskInfo, thinking2, answer2], debate_instr_stage3, r, is_sub_task=True)
            else:
                input_infos_3 = [taskInfo, thinking2, answer2] + all_thinking_stage3[r-1] + all_answer_stage3[r-1]
                thinking3, answer3 = await agent(input_infos_3, debate_instr_stage3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating candidates, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking_stage3[r].append(thinking3)
            all_answer_stage3[r].append(answer3)
    final_decision_agent_stage3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_stage3([taskInfo, thinking2, answer2] + all_thinking_stage3[-1] + all_answer_stage3[-1], "Sub-task 4: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent stage 3, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc_stage3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc_stage3)
    print("Step 3: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs
