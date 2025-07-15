async def forward_174(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction = "Sub-task 1: Analyze and classify the physical setup of the oscillating spheroidal charge distribution with symmetry axis along z-axis, identifying key parameters and missing information without assuming radiation pattern or wavelength dependence."
    cot_sc_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_1 = []
    possible_thinkings_1 = []
    subtask_desc1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking1, answer1 = await cot_sc_agents[i]([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents[i].id}, analyzing physical setup, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_answers_1.append(answer1)
        possible_thinkings_1.append(thinking1)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1([taskInfo] + possible_thinkings_1 + possible_answers_1, "Sub-task 1: Synthesize and choose the most consistent analysis of the physical setup.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    debate_instruction_2 = "Sub-task 2: Determine the lowest-order nonzero multipole moment (dipole, quadrupole, etc.) of the oscillating spheroidal charge distribution based on geometry and oscillation mode. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking2 = []
    all_answer2 = []
    subtask_desc2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": debate_instruction_2,
        "context": ["user query", thinking1, answer1.content],
        "agent_collaboration": "Debate"
    }
    for i, agent in enumerate(debate_agents_2):
        thinking2, answer2 = await agent([taskInfo, thinking1, answer1], debate_instruction_2, 0, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, round 0, determining multipole moment, thinking: {thinking2.content}; answer: {answer2.content}")
        all_thinking2.append(thinking2)
        all_answer2.append(answer2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo, thinking1, answer1] + all_thinking2 + all_answer2, "Sub-task 2: Synthesize and choose the most consistent multipole moment.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    debate_instruction_3_1 = "Sub-task 1 of Stage 2: Derive the angular dependence g(theta) of the radiation pattern from the identified multipole moment, ensuring normalization so that max_theta g(theta) = 1. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_3_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking3_1 = []
    all_answer3_1 = []
    subtask_desc3_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": debate_instruction_3_1,
        "context": ["user query", thinking1, answer1.content, thinking2, answer2.content],
        "agent_collaboration": "Debate"
    }
    for i, agent in enumerate(debate_agents_3_1):
        thinking3_1, answer3_1 = await agent([taskInfo, thinking1, answer1, thinking2, answer2], debate_instruction_3_1, 0, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, round 0, deriving angular dependence, thinking: {thinking3_1.content}; answer: {answer3_1.content}")
        all_thinking3_1.append(thinking3_1)
        all_answer3_1.append(answer3_1)
    final_decision_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3_1, answer3_1 = await final_decision_agent_3_1([taskInfo, thinking1, answer1, thinking2, answer2] + all_thinking3_1 + all_answer3_1, "Sub-task 1 of Stage 2: Synthesize and choose the most consistent angular dependence.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3.1 output: thinking - {thinking3_1.content}; answer - {answer3_1.content}")
    subtask_desc3_1['response'] = {"thinking": thinking3_1, "answer": answer3_1}
    logs.append(subtask_desc3_1)
    print("Step 3.1: ", sub_tasks[-1])

    debate_instruction_3_2 = "Sub-task 2 of Stage 2: Derive the wavelength dependence exponent n in the radiation pattern f(lambda, theta) = g(theta) * lambda^(-n) based on electromagnetic radiation theory for the identified multipole moment. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_3_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking3_2 = []
    all_answer3_2 = []
    subtask_desc3_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": debate_instruction_3_2,
        "context": ["user query", thinking1, answer1.content, thinking2, answer2.content],
        "agent_collaboration": "Debate"
    }
    for i, agent in enumerate(debate_agents_3_2):
        thinking3_2, answer3_2 = await agent([taskInfo, thinking1, answer1, thinking2, answer2], debate_instruction_3_2, 0, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, round 0, deriving wavelength dependence, thinking: {thinking3_2.content}; answer: {answer3_2.content}")
        all_thinking3_2.append(thinking3_2)
        all_answer3_2.append(answer3_2)
    final_decision_agent_3_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3_2, answer3_2 = await final_decision_agent_3_2([taskInfo, thinking1, answer1, thinking2, answer2] + all_thinking3_2 + all_answer3_2, "Sub-task 2 of Stage 2: Synthesize and choose the most consistent wavelength dependence.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3.2 output: thinking - {thinking3_2.content}; answer - {answer3_2.content}")
    subtask_desc3_2['response'] = {"thinking": thinking3_2, "answer": answer3_2}
    logs.append(subtask_desc3_2)
    print("Step 3.2: ", sub_tasks[-1])

    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction = "Sub-task 3 of Stage 2: Critically evaluate the limitations of the given information and assumptions made so far, identifying remaining uncertainties or missing data that prevent a definitive conclusion about the radiation pattern and wavelength scaling. Propose additional information or assumptions necessary to resolve these uncertainties." + reflect_inst
    cot_agent_reflect = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_reflect = [taskInfo, thinking3_1, answer3_1, thinking3_2, answer3_2]
    subtask_desc3_3 = {
        "subtask_id": "stage_2.subtask_3",
        "instruction": cot_reflect_instruction,
        "context": ["user query", thinking3_1, answer3_1.content, thinking3_2, answer3_2.content],
        "agent_collaboration": "Reflexion"
    }
    thinking3_3, answer3_3 = await cot_agent_reflect(cot_inputs_reflect, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_reflect.id}, evaluating limitations, thinking: {thinking3_3.content}; answer: {answer3_3.content}")
    for i in range(self.max_round):
        critic_inst = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
        feedback, correct = await critic_agent([taskInfo, thinking3_3, answer3_3], critic_inst, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip() == "True":
            break
        cot_inputs_reflect.extend([thinking3_3, answer3_3, feedback])
        thinking3_3, answer3_3 = await cot_agent_reflect(cot_inputs_reflect, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_reflect.id}, refining evaluation, thinking: {thinking3_3.content}; answer: {answer3_3.content}")
    sub_tasks.append(f"Sub-task 3.3 output: thinking - {thinking3_3.content}; answer - {answer3_3.content}")
    subtask_desc3_3['response'] = {"thinking": thinking3_3, "answer": answer3_3}
    logs.append(subtask_desc3_3)
    print("Step 3.3: ", sub_tasks[-1])

    debate_instruction_4 = "Stage 3 Sub-task 1: Combine the derived angular function g(theta) and wavelength dependence lambda^(-n) to calculate the fraction of maximum power radiated at theta = 30 degrees. Use this to select the correct multiple-choice option. Ensure physical consistency with multipole analysis and wavelength scaling. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking4 = []
    all_answer4 = []
    subtask_desc4 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": debate_instruction_4,
        "context": ["user query", thinking3_1, answer3_1.content, thinking3_2, answer3_2.content, thinking3_3, answer3_3.content],
        "agent_collaboration": "Debate"
    }
    for i, agent in enumerate(debate_agents_4):
        thinking4, answer4 = await agent([taskInfo, thinking3_1, answer3_1, thinking3_2, answer3_2, thinking3_3, answer3_3], debate_instruction_4, 0, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, round 0, combining results and selecting option, thinking: {thinking4.content}; answer: {answer4.content}")
        all_thinking4.append(thinking4)
        all_answer4.append(answer4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo, thinking3_1, answer3_1, thinking3_2, answer3_2, thinking3_3, answer3_3] + all_thinking4 + all_answer4, "Stage 3 Sub-task 1: Provide final answer selecting the correct multiple-choice option.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs
